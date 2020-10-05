# EPSWAE
Encoded Prior Sliced Wasserstein AutoEncoder

This work learns an prior that matches the shape of an arbitrary data manifold embedded in latent space using a prior-encoder network. A nonlinear sliced wasserstein distance between the posterior is prior distributions is minimized in the loss. It also introduces a network-geodesic algorithm to interpolate along 'network-geodesics', i.e., curves on the manifold instead of using the less natural euclidian distance. 

Full version on arxiv: https://arxiv.org/abs/2010.01037

This is a pytorch implementation and requires python 3.x

Data specific code for MNIST and CelebA is in the folders. In addition, NetworkGeodesics.py and SWAE_Empirical.py are required. NetworkGeodesics.py is an energy-based algorithm for interpolations along curved network-geodesics. SWAE_Empirical.py consists of helper function to compute the p-Sliced Wasserstein distance.

In order to run this follow these steps for a given dataset (MNIST/CelebA) :

1. Run EPSWAE_'dataset'.py: Create folders 'results' and 'models'. This saves two models per epoch (one for the autoencoder network and one for the prior encoder network) inside the models folder and saves reconstructed images in the results folder. It also saves ClusterCenters_'dataset'_EPSWAE.txt (that saves the input to the prior-encoder network). It uses SWAE_Empirical that contains utility functions needed to compute the Sliced Wasserstein distance. 

2. Then run 'dataset'_interpolation.py: This loads the models at a given epoch, and the ClustersCenter txt file, and generates network-geodesics from them. It uses Network-Geodesics.py that contains utility functions that implements the network algorithm and generates geodesics (note this uses graph data structures and requires installation of package networkx). This outputs interpolations in the results folder between two images (source and target) along the latent network-geodesic.

This work is currently under review at ICLR 2021. If you use any of the code above please cite https://arxiv.org/abs/2010.01037

