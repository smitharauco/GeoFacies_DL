# GeoFacies_DL

Development of robust parameterizations for geological facies  models. 

We present reports the current results of our investigations on the use of deep neural networks towards the construction of a continuous parameterization of facies which can be used for data assimilation with ensemble methods. 
Specifically, we use a convolutional variational autoencoder and the ensemble smoother with multiple data assimilation. 

# Result

We tested the parameterization in three synthetic history-matching problems with channelized facies.
We focus on this type of facies because they are among the most challenging to preserve after the assimilation of data. 
Our experiments were presented in the article "Towards a Robust Parameterization for Conditioning Facies Models Using Deep Variational Autoencoders and Ensemble Smoother" using a function network "DCVAE" of this repository. Also we included new variants of convolutional variational autoencoder.

# Conclusions
The parameterization showed promising results outperforming previous methods and generating well-defined channelized facies.


# References

Canchumuni, S. W., Emerick, A. A., & Pacheco, M. A. C. (2018). Towards a Robust Parameterization for Conditioning Facies Models Using Deep Variational Autoencoders and Ensemble Smoother. arXiv preprint arXiv:1812.06900. 

<a href="http://export.arxiv.org/abs/1812.06900" rel="nofollow">http://export.arxiv.org/abs/1812.06900</a>

Emerick, A. A., & Reynolds, A. C. (2013). Ensemble smoother with multiple data assimilation. Computers & Geosciences.

<a href="http://export.arxiv.org/abs/1812.06900" rel="nofollow">
https://www.sciencedirect.com/science/article/pii/S0098300412000994</a>

