#Utility functions for building and analyzing graphs and network-geodesics
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
from sklearn.decomposition import PCA
import SWAE_Emperical as swae 
import scipy.io as sio
import networkx as nx


#assembles a weighted directed graph in which distances are computed pairwise between all latent samples
def GenGraph(nnbs,samples,thresh=1.0,device='cpu',p=1):
    """generates a thresholded network from samples
        Args:
            samples
            nnbs (int): number of neighbors used in determining centrality
            thresh (float): fraction of centrality used to threshold edges
            thresh_inc (float): change in threshold in each step
            p: power of edge weight (default = 1) 

        Return:
            eweights (Tensor) : adjacency matrix of graph
    """
            
    ns = shape(samples)[0]
    eweights = torch.zeros(ns,ns).to(device)
    dists = torch.zeros(ns,ns).to(device)
    dists = torch.cdist(samples.unsqueeze(0),samples.unsqueeze(0)).squeeze()
    tadist = torch.sum(dists)/(ns**2) #average distance between all the points
    print(tadist,'total average dist')
    odists,_ = torch.sort(dists,dim=1,descending=False)
    #dentify the threshold for each node (sample)
    centrality = torch.zeros(ns).to(device) 
    for k in range(ns):
        adists = 0 
        for j in range(nnbs):
            adists += odists[k,1+j]
        centrality[k] = adists/nnbs #centrality is the average distance of the nnbs-nearest neighbors 
    tcons = 0

    #threshold the network
    for k in range(ns):
        for j in range(ns):
            #threshold the edge weights. remove edges between far away points
            if dists[k,j] > thresh*centrality[k]:
                eweights[k,j]=0
            else:
                tcons = tcons + 1

                eweights[k,j] = torch.pow(dists[k,j],p)
    print(tcons,'total connections')
    print(ns**2,'total possible connections')
                    
    return eweights #gives the weighted adjacency matrix 

            


#Keep increasing the threshold until the resulting graph is connected 
def GenConnectedNetwork(samples,nnbs,thresh=1.0,thresh_inc=.25,p=1):
    """generates a thresholded network from samples
        Args:
            samples
            nnbs (int): number of neighbors used in determining centrality
            thresh (float): fraction of centrality used to threshold edges
            thresh_inc (float): change in threshold in each step
            p: power of edge weight (default = 1) 

        Return:
            G (graph): thresholded graph of samples
    """
    connected = False
    while(connected == False):
        ng = GenGraph(nnbs,samples,thresh=thresh,p=p)
        G = nx.DiGraph(ng.detach().cpu().numpy())
        if(nx.is_strongly_connected(G)): connected=True
        else:
            thresh = thresh + thresh_inc
            print('increasing threshold because graph is not connected!')
    return G

#generates a single interpolation for a given source and target and returns the metric distance, given a graph G and two nodes of the graph
def GenInterpolation(G,source,target,p=1):
    """identify shortest path and along allowed path using Djikstra's algorithm (or some variant thereof)
        Args:
            G (graph): graph of samples
            source (int): start point of network-geodesic
            target (int): end point of network-geodesic
            p: power of edge weight (default = 1) 

        Return:
            geodesic distance (int) , path (list)
    """
    d = nx.dijkstra_path(G,source,target)
    length = shape(d)[0]
    mdist = 0
    for k in range(length-1):
        mdist += G[d[k]][d[k+1]]['weight']
        
    return d, np.power(mdist,1.0/p) 

#returns all network-geodesics between start point i and end point j
def PairwiseGeodesics(G,n,p=1.0):
    print(n, 'number of nodes')
    geo_dist = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            path = nx.dijkstra_path(G,i,j)
            tot=0
            for k in range(len(path)-1):
                tot += G[path[k]][path[k+1]]['weight']
            geo_dist[i,j] = geo_dist[j,i] = pow(tot,1.0/p)
    return geo_dist
