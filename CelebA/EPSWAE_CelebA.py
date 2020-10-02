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
import SWAE_Emperical as swae 


parser = argparse.ArgumentParser(description='PEVAE CelebA Example')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 200)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval',type=int, default=100,help='how many epochs to wait before saving the models')
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
parser.add_argument('--ramp-beta',default=False,help='schedules the regularization to increase gradually to the specified regularization') 
parser.add_argument('--wass-p',type=int,default=2,help='sets the lp norm used to compute Wasserstein norm')
parser.add_argument('--nprojs',type=int,default=250,help='sets the number of projections used to compute sliced Wasserstein distance')
parser.add_argument('--start-epoch',type=int,default=0,help='what epoch is this on restart')
parser.add_argument('--alpha',type=float,default =.001, help='hyperparameter for structure-consistency loss')
parser.add_argument('--clus-sep',type=float,default =2.0, help='hyperparameter cluster sep')

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
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


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
        basesamples = SampleGMM_detach(nsamps) #.to(device) 
        return self.forward(basesamples)
    
    def GenNsamples_withPrior(self,nsamps):
        basesamples = SampleGMM_detach(nsamps) #.to(device) 
        return self.forward(basesamples),basesamples

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

    def encode_FE(self, x):
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
        return self.mlp3(x),fe

        
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
        return self.mlp3(x),fe

    

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
        mu,fe = self.encode(x)
        return self.decode(mu), mu

    def forward_FE(self,x):
        mu,fe = self.encode_FE(x)        
        return self.decode(mu), mu,fe

###################################################    

def SampleGMM_detach(nsamps): 
    sample = gmm.sample((nsamps,))
    return sample

def FSCloss(data,mu):
    bs = shape(data)[0]
    pd_data = torch.pow(torch.cdist(data.detach(),data.detach()),2)
    pd_mu = torch.pow(torch.cdist(mu,mu,compute_mode='donot_use_mm_for_euclid_dist'),2)
    ad_data = torch.sum(pd_data.detach())/(bs**2)
    ad_mu = torch.sum(pd_mu)/(bs**2)      
    wtf = torch.pow(torch.log(1 + pd_data/ad_data) - torch.log(1 + pd_mu/ad_mu),2)
    loss = torch.sum(wtf)/bs
    return loss

def shear(emp,kvec1,kvec2,device=device):
    bs,ld = shape(emp)
    trnsfrm = torch.zeros(bs,ld,device=device)

    kvec1=kvec1.unsqueeze(1)
    kvec2=kvec2.unsqueeze(1)

    trnsfrm = emp + torch.matmul(torch.sin(2*np.pi*torch.matmul(emp,kvec2)) , kvec1.T)
    return trnsfrm

def SWDnonlin(emp,sampler,nnproj=5,nlproj=50,device=device):
    bs,ld = shape(emp)
    swd = torch.zeros(nnproj,device=device)
    swd[0] = swae.sliced_wasserstein_distance(emp,sampler,num_projections=nlproj,p=args.wass_p,device=device)
    for j in range(1,nnproj):
        std = torch.std(emp)
        kvec1 = std*torch.randn(ld,device=device)
        kvec2 = std*torch.randn(ld,device=device)
        swd[j] = swae.sliced_wasserstein_distance( shear(emp,kvec1,kvec2,device=device), lambda x: shear(sampler(x),kvec1,kvec2,device=device) ,num_projections=nlproj,p=args.wass_p,device=device)
    return torch.mean(swd)


################################################################

def loss_function(recon_batch, data, mu, fe):
    BCE = F.binary_cross_entropy(recon_batch, data.view(-1, int(leng*leng)), reduction='mean')
    SWD=SWDnonlin(mu,prior.GenNsamples_NormPrior,device=device)

    fscloss = FSCloss(fe,mu)
    return BCE + beta*SWD + args.alpha*fscloss

    
def prior_loss(data,mu):
    SWD=SWDnonlin(mu,prior.GenNsamples_NormPrior) 
    return SWD 

###############################################################


#autoencoder
model = AE(Z).to(device)
#prior-encoder
prior = PriorE(P,Z).to(device)

optimizer = optim.Adam(model.parameters(), lr=lrm)
prior_optimizer = optim.Adam(prior.parameters(), lr=lrp)
#################################################


def train(epoch):
    model.train()
    prior.train()
    train_loss = 0
    global beta    
    for batch_idx, (data,y) in enumerate(train_loader):
        if(batch_idx>5):break
        #One step of Adam for the AE
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, fe = model.forward_FE(data)        
        print ( 'epoch ', epoch ,', batch ',batch_idx, )
        loss = loss_function(recon_batch, data, mu, fe)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        for k in range(args.pessteps):
            prior_optimizer.zero_grad()
            mu2,_ = model.encode(data) 
            lossp = prior_loss(data,mu2)
            lossp.backward()
            prior_optimizer.step()

        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    return recon_batch, data



if __name__ == "__main__":
    global mu_pr1
    global var_pr1

    #initialize inputs to prior-encoder
    mu_pr1 = torch.zeros(nclus,P,device=device) + args.clus_sep*torch.randn(nclus,P,device=device)
    var_pr1= torch.zeros(nclus,P,P,device=device)
    #store centers of gaussian inputs to prior-encoder
    clscenlog = 'ClusterCenters_CELEBA_EPSWAE.txt'
    for i in range(nclus):
        var_pr1[i]=torch.eye(P,device=device).detach()
    np.savetxt(clscenlog,mu_pr1.detach().numpy())
        
    alpha_pr = torch.zeros(nclus,device=device)
    for k in range(nclus): alpha_pr[k] = 1.0/nclus

    mix = distributions.Categorical(alpha_pr)
    comp = distributions.MultivariateNormal(mu_pr1,var_pr1)
    gmm = distributions.MixtureSameFamily(mix,comp)

        
    for epoch in range(1, args.epochs + 1):
        recon_batch, data = train(epoch)
        
        if(epoch >0 and epoch%args.save_interval == 0):
            torch.save(model.state_dict(),'models/CELEBA_EPSWAE_AE_e' + str(epoch))
            torch.save(prior.state_dict(),'models/CELEBA_EPSWAE_PE_e' + str(epoch))
                

        #generate sample plots
        sample = prior.GenNsamples_NormPrior(64) 
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 3, leng,leng),'results/CelebA_EPSWAE_Sample_e_' + str(epoch) + '.png')
        close('all')

        n = min(data.size(0), 8)
        comparison = torch.cat([data.to(device)[:n],recon_batch.to(device).view(-1, 3, leng,leng)[:n]])
        save_image(comparison.cpu(),'results/CelebA_EPSWAE_Reconstruction_e_' + str(epoch) + '.png', nrow=n)

        


