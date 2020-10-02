#Load model and generate interpolations for CelebA
#Copyright © 2020 Sanjukta Krishnagopal & Jacob Bedrossian

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim, distributions
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
from numpy import *
from matplotlib.pyplot import *
from sklearn.manifold import TSNE
import sklearn
import networkx as nx
import NetworkGeodesics as nic

parser = argparse.ArgumentParser(description='PEVAE CelebA Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval',type=int, default=50,help='how many epochs to wait before saving the models')
parser.add_argument('--nclus', type=int, default=10, metavar='N',
                    help='how many gaussians in the mixture in prior space')
parser.add_argument('--latentdim', type=int, default=10, metavar='N',
                    help='dimension of latent space')
parser.add_argument('--priordim',type=int,default=186, help='dimension of the prior space')
parser.add_argument('--restart', default=False,help='loads the NNs from file and restarts training; must specific model and prior encoder with model and pe flags')
parser.add_argument('--model',type=str,default='',help='file holding the state dictionary of the autoencoder')
parser.add_argument('--pe',type=str,default='',help='file holding the state dictionary of the prior encoder')
parser.add_argument('--lrm',type=float,default=1e-3,help='learning rate for the autoencoder')
parser.add_argument('--lrp',type=float,default=1e-3,help='learning rate for the prior encoder')
parser.add_argument('--pessteps',type=int,default=1,help='how many steps of prior training per AE training')
parser.add_argument('--beta',type=float,default=1, help='regularization parameter in AE loss')
parser.add_argument('--wass-p',type=int,default=2,help='sets the lp norm used to compute Wasserstein norm')
parser.add_argument('--nprojs',type=int,default=250,help='sets the number of projections used to compute sliced Wasserstein distance')
parser.add_argument('--gen-plots',default=True,help='flags whether or not the code will generate the standard array of plots each epoch')
parser.add_argument('--save-flag',type=str,default='',help='specifies a unique ID for the run. Otherwise a random ID is generated')
parser.add_argument('--clus-sep',type=float,default=3,help='specifies how far apart the prior clusters are')
parser.add_argument('--start-epoch',type=int,default=1,help='what epoch is this on restart')
parser.add_argument('--scloss',default=False,help='turn on the structure-consistency loss')
parser.add_argument('--alpha',type=float,default =.001, help='hyperparameter for structure-consistency loss')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
nclus=args.nclus
latentdim=args.latentdim
P = args.priordim
Z = args.latentdim
bs = args.batch_size
lrm = args.lrm
lrp = args.lrp
beta = args.beta

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


#64x64 downsized images 
leng_h = 64
leng_w = 64
leng=64

def get_celeba(batchSize):
    data_path='../data'

    train_transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    train_data = datasets.ImageFolder(data_path,train_transformation)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batchSize,
        shuffle=True,**kwargs)
    return train_loader


train_loader = get_celeba(args.batch_size)

global mu_pr1
global var_pr1

class PriorE(nn.Module):
    def __init__(self,L1,L2):
        super(PriorE, self).__init__()

        self.fc1 = nn.Linear(L1,L1)
        self.fc2 = nn.Linear(L1,L1)
        self.fc3 = nn.Linear(L1,L1)
        self.fc4 = nn.Linear(L1,L2)

        self.L1 = L1
        self.L2 = L2

    def forward(self,z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        return self.fc4(z)

    def GenNsamples_NormPrior(self,nsamps):
        basesamples = SampleGMM_detach(nsamps).to(device) 
        return self.forward(basesamples)

class AE(nn.Module):
    def __init__(self,Z):
        super(AE, self).__init__()

        #This will be the encoder
        self.conv1 = nn.Conv2d(3,16,3,padding=1) #will reduce to 64x64
        self.pool = nn.MaxPool2d(2,2) #Will reduce to 32 x 32
        self.conv2 = nn.Conv2d(16,32,3,padding=1) #Will reduce to 32x32
        #pool again will reduce further to 16x16
        self.conv3 = nn.Conv2d(32,64,3,padding=1) 

            
        self.mlp1 = nn.Linear(8*8*64, 512) 
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, Z)

        #The decoder will be the MLP in reverse)
        self.mlp1d = nn.Linear(Z, 256) 
        self.mlp2d = nn.Linear(256, 512) 
        self.mlp3d = nn.Linear(512,8*8*64)

        self.dconv1 = nn.ConvTranspose2d(64,32,3,padding=1)
        self.upsamp = nn.Upsample(scale_factor=2)
        self.dconv2 = nn.ConvTranspose2d(32,16,3,padding=1)
        self.dconv3 = nn.ConvTranspose2d(16,3,3,padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)


    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1,8*8*64)
        fe = x
        x = F.leaky_relu(self.mlp1(x))
        x = F.leaky_relu(self.mlp2(x))
        return self.mlp3(x)

    def decode(self, z):
        z = F.leaky_relu(self.mlp1d(z))
        z = F.leaky_relu(self.mlp2d(z))
        z = F.leaky_relu(self.mlp3d(z))

        z = z.view(-1,64,8,8)
        z = self.upsamp(z)
        z = F.leaky_relu(self.dconv1(z))
        z = self.upsamp(z)
        z = F.leaky_relu(self.dconv2(z))
        z = self.upsamp(z)
        z = self.dconv3(z)
        z = torch.sigmoid(z)
        return z.view(-1,int(leng*leng))

    def forward(self, x):
        mu = self.encode(x)
        return self.decode(mu), mu


###################################################
#generate prior samples
def SampleGMM_detach(nsamps): 
    global mu_pr1
    global var_pr1

    Z = shape(mu_pr1)[1]
    K = shape(mu_pr1)[0]
    alpha_pr = torch.zeros(K)
    for k in range(K): alpha_pr[k] = 1.0/K

    mix = distributions.Categorical(alpha_pr)
    comp = distributions.MultivariateNormal(mu_pr1.detach(),var_pr1.detach())
    #I have no idea if that works
    gmm = distributions.MixtureSameFamily(mix,comp)

    sample=torch.zeros(nsamps,Z).to(device)
    sample = gmm.sample((nsamps,))
    return sample


###################################################
       

if __name__ == "__main__":
    model = AE(Z).to(device)
    prior = PriorE(P,Z).to(device)

    modelfln = 'models/CELEBA_EPSWAE_AE_e' + str(args.start_epoch)
    pefln = 'models/CELEBA_EPSWAE_PE_e' + str(args.start_epoch)
    
    if(not args.cuda):
        model.load_state_dict(torch.load(modelfln,map_location=torch.device('cpu')))
        model.eval()
        prior.load_state_dict(torch.load(pefln,map_location=torch.device('cpu')))
        prior.eval()
    else:
        model.load_state_dict(torch.load(modelfln))
        model.eval()
        prior.load_state_dict(torch.load(pefln))
        prior.eval()


    clscenlog = 'ClusterCenters_CELEBA_EPSWAE.txt'

        
    global mu_pr1
    global var_pr1
    var_pr1= torch.zeros(nclus,P,P)
    for i in range(nclus):
        var_pr1[i]=torch.eye(P).detach()
    mu_prnp = np.loadtxt(clscenlog)
    mu_pr1 = torch.from_numpy(mu_prnp).float().to(device)
        
    #tests generation for visual inspection
    sample = prior.GenNsamples_NormPrior(64)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 3, leng,leng),'results/CelebA_Samples_e_' + str(args.start_epoch) + '.png')
    close('all')

    
    data,y = next(iter(train_loader))
    recon,mu_last = model(data)
    bs = shape(data)[0]
    
    figure()
    mugen = prior.GenNsamples_NormPrior(bs)         
    #posterior under TSNE
    close('all')

    mup = torch.cat([mu_last,mugen],axis=0)
    
    #####Now we generate sample interpolations
    G = nic.GenConnectedNetwork(mup,int(0.25*bs),thresh=1.0,thresh_inc=.1,p=1)

    print('Network finished')

    
    #plot some random interpolations
    for k in range(10):
        source = random.randint(bs)
        target = random.randint(bs)
        d, geodist = nic.GenInterpolation(G,source,target)
        sample = model.decode(mup[d]).cpu()

        #plot interpolations along network-geodesic
        figure()
        save_image(sample.view(len(d), 3, leng,leng),'results/CelebA_geodesic_interp_Z' + str(Z) + '_e_' + str(args.start_epoch) + '_i_' + str(k) + '.png',nrow=len(d))
        close('all')

        path = torch.zeros(2*len(d)-1,Z)
        #Now let's smooth the interpolation by adding points along the linear interpolation
        for j in range(len(d)-1):
            path[2*j] = mup[d[j]]
            path[2*j+1] = 0.5*mup[d[j]] + 0.5*mup[d[j+1]]
        path[-1] = mup[d[-1]]
            
        rsample=model.decode(path).cpu()
        figure()
        save_image(rsample.view(-1, 3, leng,leng),'results/CelebA_manifold_interpolated_Z' + str(Z) + '_e_' + str(args.start_epoch) + '_i_' + str(k) + '.png',nrow=len(path))
        close('all')

