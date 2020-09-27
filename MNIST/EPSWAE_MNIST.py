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
from sklearn.cluster import KMeans
from numpy import *
from matplotlib.pyplot import *
from sklearn.manifold import TSNE
import SWAE_Emperical as swae 


parser = argparse.ArgumentParser(description='EPSWAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval',type=int, default=10,help='how many epochs to wait before saving the models')
parser.add_argument('--nclus', type=int, default=10, metavar='N',
                    help='how many gaussians in the mixture in prior space')
parser.add_argument('--latentdim', type=int, default=10, metavar='N',
                    help='dimension of latent space')
parser.add_argument('--priordim',type=int,default=40, help='dimension of the prior space')
parser.add_argument('--model',type=str,default='',help='file holding the state dictionary of the autoencoder')
parser.add_argument('--pe',type=str,default='',help='file holding the state dictionary of the prior encoder')
parser.add_argument('--lrm',type=float,default=1e-3,help='learning rate for the autoencoder')
parser.add_argument('--lrp',type=float,default=1e-3,help='learning rate for the prior encoder')
parser.add_argument('--pessteps',type=int,default=1,help='how many steps of prior training per AE training')
parser.add_argument('--beta',type=float,default=1, help='regularization parameter in AE loss')
parser.add_argument('--wass-p',type=int,default=2,help='sets the lp norm used to compute Wasserstein norm')
parser.add_argument('--nprojs',type=int,default=75,help='sets the number of projections used to compute sliced Wasserstein distance')
parser.add_argument('--start-epoch',type=int,default=0,help='what epoch is this on restart')
parser.add_argument('--alpha',type=float,default =.001, help='hyperparameter for structure-consistency loss')
parser.add_argument('--clus-sep',type=float,default =2.0, help='hyperparameter cluster sep')


global mu_pr1
global var_pr1

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
nclus=args.nclus
latentdim=args.latentdim
P = args.priordim
Z = args.latentdim
bs = args.batch_size
lrm = args.lrm
lrp = args.lrp

print('learning range model',lrm)
print('learning range model',lrp)
beta = args.beta

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
d,y = next(iter(train_loader))
_,_,leng,_ =shape(d)



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

        #Encoder layers
        self.conv1 = nn.Conv2d(1,10,3) #will reduce to 26x26
        self.pool = nn.MaxPool2d(2,2) #Will reduce to 13 x 13
        self.conv2 = nn.Conv2d(10,16,3) #Will reduce to 11x11

        self.mlp1 = nn.Linear(11*11*16, 512) 
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, Z)

        #Decoder layers
        self.mlp1d = nn.Linear(Z, 256) 
        self.mlp2d = nn.Linear(256, 512) 
        self.mlp3d = nn.Linear(512,11*11*16)

        self.dconv1 = nn.ConvTranspose2d(16,10,3)
        self.upsamp = nn.Upsample(scale_factor=2)
        self.dconv2 = nn.ConvTranspose2d(10,1,3)

        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(16)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x) 
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(-1,11*11*16)
        fe = x
        
        x = F.leaky_relu(self.mlp1(x))
        x = F.leaky_relu(self.mlp2(x))
        return self.mlp3(x), fe 

    def decode(self, z):
        z = F.leaky_relu(self.mlp1d(z))
        z = F.leaky_relu(self.mlp2d(z))
        z = F.leaky_relu(self.mlp3d(z))

        z = z.view(-1,16,11,11)
        z = self.dconv1(z)
        z = F.leaky_relu(z)
        z = self.upsamp(z)
        z = torch.sigmoid(self.dconv2(z))
        return z.view(-1,int(leng*leng))

    
    def forward(self, x):
        mu,fe = self.encode(x)        
        return self.decode(mu), mu, fe


################################################################

# generate samples to input to prior encoder network
def SampleGMM_detach(nsamps): 
    global mu_pr1
    global var_pr1

    Z = shape(mu_pr1)[1]
    K = shape(mu_pr1)[0]
    alpha_pr = torch.zeros(K)
    for k in range(K): alpha_pr[k] = 1.0/K

    mix = distributions.Categorical(alpha_pr)
    comp = distributions.MultivariateNormal(mu_pr1.detach(),var_pr1.detach())
    gmm = distributions.MixtureSameFamily(mix,comp)
    sample=torch.zeros(nsamps,Z).to(device)
    sample = gmm.sample((nsamps,))

    return sample

#compute feature structural consistency (FSC) loss
def FSCloss(data, mu):
    bs = shape(data)[0]
    pd_data = torch.pow(torch.cdist(data.detach(),data.detach()),2)
    pd_mu = torch.pow(torch.cdist(mu,mu,compute_mode='donot_use_mm_for_euclid_dist'),2)
    ad_data = torch.sum(pd_data.detach())/(bs**2)
    ad_mu = torch.sum(pd_mu)/(bs**2)      
    wtf = torch.pow(torch.log(1 + pd_data/ad_data) - torch.log(1 + pd_mu/ad_mu),2)
    loss = torch.sum(wtf)/bs
    return loss

#shear samples in latent space nonlinearly for the NSW distance
def shear(emp,kvec1,kvec2,device=device):
    bs,ld = shape(emp)
    trnsfrm = torch.zeros(bs,ld).to(device)

    #direction of shear
    kvec1=kvec1.unsqueeze(1).to(device)
    #frequency of shear
    kvec2=kvec2.unsqueeze(1).to(device)

    trnsfrm = emp + torch.matmul(torch.sin(2*np.pi*torch.matmul(emp,kvec2)) , kvec1.T)
    return trnsfrm

# calculate the nonlinear sliced wasserstein distance
def SWDnonlin(emp,sampler,nnproj=5,nlproj=50,device=device):
    bs,ld = shape(emp)
    swd = torch.zeros(nnproj).to(device)
    swd[0] = swae.sliced_wasserstein_distance(emp,sampler,num_projections=nlproj,p=args.wass_p,device=device)
    for j in range(1,nnproj):
        kvec1 = torch.var(emp)*torch.randn(ld).to(device)
        kvec2 = torch.randn(ld).to(device)
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

################################################################
def train(epoch):
    model.train()
    prior.train()
    train_loss = 0
    global beta
    for batch_idx, (data, y) in enumerate(train_loader):
        #One step of Adam for the AE 
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, fe = model(data)
        print ( 'epoch ', epoch ,', batch ',batch_idx, )
        loss = loss_function(recon_batch, data, mu, fe)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        #args.pessteps steps of Adam for the prior
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

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))


    #Save models each epoch
    if(epoch > args.start_epoch and epoch % args.save_interval == 0):
        torch.save(model.state_dict(),'models/MNIST_EPSWAE_AE_e' + str(epoch))
        torch.save(prior.state_dict(),'models/MNIST_EPSWAE_PE_e' + str(epoch))
        
    return mu2, y

#testing - reconstruction
def test(epoch):
    model.eval()
    prior.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu,fe = model(data)
            loss = loss_function(recon_batch, data, mu,fe)
            test_loss+=loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, leng, leng)[:n]])
                save_image(comparison.cpu(),
                           'results/MNIST_EPSWAE_reconstruction _e_' + str(epoch) + '.png', nrow=n)

        print('====> Test set loss: {:.4f}'.format(test_loss))


        
if __name__ == "__main__":
    global mu_pr1
    global var_pr1


    #initialize inputs to prior-encoder
    mu_pr1 = torch.zeros(nclus,P) + args.clus_sep*torch.randn(nclus,P)
    var_pr1= torch.zeros(nclus,P,P)
    #store centers of gaussian inputs to prior-encoder
    clscenlog = 'ClusterCenters_MNIST_EPSWAE.txt'
    for i in range(nclus):
        var_pr1[i]=torch.eye(P).detach()
        np.savetxt(clscenlog,mu_pr1.detach().numpy())

    for epoch in range(args.start_epoch+1, args.epochs + args.start_epoch + 1):
        mu_last, y = train(epoch)
        test(epoch)
