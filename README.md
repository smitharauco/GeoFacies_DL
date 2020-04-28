# GeoFacies_DL

**Development of robust parameterizations for geological facies  models.**

We present reports the current results of our investigations on the use of deep neural networks towards the construction of a continuous parameterization of facies which can be used for data assimilation with ensemble methods. 
Specifically, we use a generative networks and the ensemble smoother with multiple data assimilation. 

* [**Convolutional Variational Autoencoders**]
* [**Convolutional Variational Autoencoders with Style Loss**]
* [**Generative adversarial networks with Encoder**]
* [**Generative adversarial networks with Encoder and Wasserstein Loss**]
* [**Auto-Encoding Generative Adversarial Networks (AlphaGans)**]
* [**Cycle GANs With PCA**]

## DataSet

More details of the Facies models dataSet can be found in the reference Paper.
To download out DataSet in TFrecord File. 

```python
from Model.Utils import download_tfrecord
download_tfrecord(name_dataset = "MPS45", test = True)
```
where: 
'name_dataset' represent the name of the DataSet

Please contact us to has any problem to download the dataset.

## Train Model
Please see python train_geofacies.py -h for optional arguments.
  use 'save_path_weights' to save all trained weights

### Train Convolutional Variational Autoencoders
```bash
# train model using a reservoir field of (45,45)
python train_geofacies.py --train_dataset_path DataSet/MPS45/train.tfrecords --test_dataset_path DataSet/MPS45/test_val.tfrecords --filters 64-32-32 --strides_values 2-2-1 --kernel_dim 3-3-3 --hidden_dim 1024 --latent_dim 500 --save_path_weights CVAE-MPS45.hdf5

# train model using a reservoir field of (60,60)
python train_geofacies.py --train_dataset_path DataSet/MPS60/train.tfrecords --test_dataset_path DataSet/MPS60/test_val.tfrecords --filters 128-64-32 --strides_values 2-2-1 --kernel_dim 3-3-3 --hidden_dim 2048 --latent_dim 500

# train model using a reservoir field of (100,100)
python train_geofacies.py --train_dataset_path DataSet/MPS100/train.tfrecords --test_dataset_path DataSet/MPS100/test_val.tfrecords --filters 128-32-16 --strides_values 2-2-2 --kernel_dim 5-5-3 --hidden_dim 2048 --latent_dim 500 --lr 0.0001

# train model using a reservoir field of (40,200)
python train_geofacies.py --train_dataset_path DataSet/MPS40x200/train.tfrecords --test_dataset_path DataSet/MPS40x200/test_val.tfrecords --filters 128-64-32 --strides_values 2-2-2 --kernel_dim 7-5-3 --hidden_dim 2048 --latent_dim 1024 --lr 0.0001 --dropout 0.0 
```
### Train Convolutional Variational Autoencoders with Style Loss
```bash
# train model using a reservoir field of (60,60)
python train_geofacies.py --train_dataset_path DataSet/MPS60/train.tfrecords --test_dataset_path DataSet/MPS60/test_val.tfrecords --filters 128-64-32 --strides_values 2-2-1 --kernel_dim 3-3-3 --hidden_dim 2048 --latent_dim 500 --lr 0.0001 --model cvae-style
```
### Train Generative adversarial networks with Encoder
```bash
# train model using a reservoir field of (60,60)
python train_geofacies.py --train_dataset_path DataSet/MPS60/train.tfrecords --test_dataset_path DataSet/MPS60/test_val.tfrecords --filters 32 --model GAN2D_AE --kernel_dim 4
# train model using a reservoir field of (40,200)
python train_geofacies.py --train_dataset_path DataSet/MPS40x200/train.tfrecords --test_dataset_path DataSet/MPS40x200/test_val.tfrecords --model GAN2D_AE --filters 32 --latent_dim 1024 --kernel_dim 5 --batch_size 64
``` 
### Train Generative adversarial networks with Encoder and Wasserstein Loss
```bash 
# train model using a reservoir field of (60,60)
python train_geofacies.py --train_dataset_path DataSet/MPS60/train.tfrecords --test_dataset_path DataSet/MPS60/test_val.tfrecords --filters 32 --model WGAN2D_AE --kernel_dim 4
# train model using a reservoir field of (40,200)
python train_geofacies.py --train_dataset_path DataSet/MPS40x200/train.tfrecords --test_dataset_path DataSet/MPS40x200/test_val.tfrecords --model WGAN2D_AE --filters 32--latent_dim 1024 --kernel_dim 5 --batch_size 64
``` 
### Train Auto-Encoding Generative Adversarial Networks (AlphaGans)
```bash 
# train model using a reservoir field of (60,60)
python train_geofacies.py --train_dataset_path /DataSet/MPS60/train.tfrecords --test_dataset_path DataSet/MPS60/test_val.tfrecords --model AlphaGAN --kernel_dim 5 --alpha 100 
# train model using a reservoir field of (40,200)
python train_geofacies.py --train_dataset_path DataSet/MPS40x200/train.tfrecords --test_dataset_path DataSet/MPS40x200/test_val.tfrecords --latent_dim 1024 --model AlphaGAN  --alpha 100 --kernel_dim 5

``` 
### Train Cycle GANs
```bash 
# train model using a reservoir field of (60,60)
python train_geofacies.py --train_dataset_path DataSet/MPS60/train.tfrecords --test_dataset_path DataSet/MPS60/test_val.tfrecords --filters 32 --Nr 3000 --Nt 3000 --model CycleGAN --kernel_dim 5 --batch_size 2 --epsilon 0.5

# train model using a reservoir field of (40,200)
python train_geofacies.py --train_dataset_path DataSet/MPS40x200/train.tfrecords --test_dataset_path DataSet/MPS40x200/test_val.tfrecords --filters 32 --Nr 5000 --Nt 5000 --model CycleGAN  --epsilon 0.5 --batch_size 2 --kernel_dim 5
``` 

## Result

We tested the parameterization in three synthetic history-matching problems with channelized facies.
We focus on this type of facies because they are among the most challenging to preserve after the assimilation of data. 
Our results were presented in the article "Towards a Robust Parameterization for Conditioning Facies Models Using Deep Variational Autoencoders and Ensemble Smoother", using the DCVAE function of this repository.


## Conclusions
The parameterization showed promising results outperforming previous methods and generating well-defined channelized facies.


## References


```tex
@article{CANCHUMUNI201987,
title = "Towards a robust parameterization for conditioning facies models using deep variational autoencoders and ensemble smoother",
journal = "Computers & Geosciences",
volume = "128",
pages = "87 - 102",
year = "2019",
issn = "0098-3004",
doi = "https://doi.org/10.1016/j.cageo.2019.04.006",
url = "http://www.sciencedirect.com/science/article/pii/S0098300419300378",
author = "Smith W.A. Canchumuni and Alexandre A. Emerick and Marco Aurélio C. Pacheco",
}
```
<a href="https://www.sciencedirect.com/science/article/pii/S0098300419300378" rel="nofollow">https://www.sciencedirect.com/science/article/pii/S0098300419300378</a>


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



