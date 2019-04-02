# GeoFacies_DL

Development of robust parameterizations for geological facies  models. 

We present reports the current results of our investigations on the use of deep neural networks towards the construction of a continuous parameterization of facies which can be used for data assimilation with ensemble methods. 
Specifically, we use a convolutional variational autoencoder and the ensemble smoother with multiple data assimilation. 

# Result

We tested the parameterization in three synthetic history-matching problems with channelized facies.
We focus on this type of facies because they are among the most challenging to preserve after the assimilation of data. 
Our results were presented in the article "Towards a Robust Parameterization for Conditioning Facies Models Using Deep Variational Autoencoders and Ensemble Smoother", using the DCVAE function of this repository.


# Conclusions
The parameterization showed promising results outperforming previous methods and generating well-defined channelized facies.


# References


```tex
@article{canchumuni2018towards,
  title={Towards a Robust Parameterization for Conditioning Facies Models Using Deep Variational Autoencoders and Ensemble Smoother},
  author={Canchumuni, Smith WA and Emerick, Alexandre A and Pacheco, Marco Aur{\'e}lio C},
  journal={arXiv preprint arXiv:1812.06900},
  year={2018}
}
```
<a href="http://export.arxiv.org/abs/1812.06900" rel="nofollow">http://export.arxiv.org/abs/1812.06900</a>


```tex
@article{emerick2013ensemble,
  title={Ensemble smoother with multiple data assimilation},
  author={Emerick, Alexandre A and Reynolds, Albert C},
  journal={Computers \& Geosciences},
  volume={55},
  pages={3--15},
  year={2013},
  publisher={Elsevier}
}
```
<a href="https://www.sciencedirect.com/science/article/pii/S0098300412000994" rel="nofollow">https://www.sciencedirect.com/science/article/pii/S0098300412000994</a>



