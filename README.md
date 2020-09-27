# EPSWAE
Encoded Prior Sliced Wasserstein AutoEncoder

This work learns a latent representation (through a prior) that encodes the data manifold using a nonlinear sliced wasserstein distance in the loss and a prior-encoder network. It also introduces a network-geodesic algorithm to interpolate along 'network-geodesics', i.e., curves on the manifold instead of using the less natural euclidian distance. 

In order to run this follow these steps for a given dataset (MNIST/CelebA) :

1. Run EPSWAE_'dataset'.py: This saves two models per epoch (one for the autoencoder network and one for the prior encoder network). It also saves ClusterCenters_'dataset'_EPSWAE.txt (that saves the input to the prior-encoder network). It uses SWAE_Empirical that contains utility functions needed to compute the Sliced Wasserstein distance. 
2. Then run 'dataset'_interpolation.py: This loads the models at a given epoch, and the ClustersCenter txt file, and generates network-geodesics from them. It uses Network-Geodesics.py that contains utility functions that implements the network algorithm and generates geodesics (note this uses graph data structures and requires installation of package networkx). This outputs interpolations between two images along the latent network-geodesic.
