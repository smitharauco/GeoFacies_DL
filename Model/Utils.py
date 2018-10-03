import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import time
from keras import backend as K
from keras.datasets import mnist
from plotly import tools
import scipy.io as sio
import matplotlib.pyplot as plt
EPSILON = 1e-8

def PlotHistory(history):
    leg=[]
    for key in history.keys():
        plt.plot(history[key])
        plt.title('Training')
        plt.xlabel('epoch') 
        leg.append(key)
    plt.legend(leg, loc='upper left')

def PlotDataAE(X,X_AE,digit_size=28,cmap='jet',Only_Result=True):
    plt.figure(figsize=(digit_size*0.5,digit_size*0.5))
    if (Only_Result):
        for i in range(0,20):
            plt.subplot(10,10,(i+1))
            plt.imshow(np.squeeze(X[i].reshape(digit_size,digit_size,1)),cmap=cmap)
            plt.axis('off')
            plt.title('Input')
        plt.show()
    plt.figure(figsize=(digit_size*0.5,digit_size*0.5))
    for i in range(0,20):
        plt.subplot(10,10,(i+1))
        plt.imshow(np.squeeze(X_AE[i].reshape(digit_size,digit_size,1)),cmap=cmap) 
        plt.axis('off')
        plt.title('Output')
    plt.show()  



def LoadMPS45(dirBase='/work/Home89/PythonUtils/DataSetThesis/MPS45.mat'):
    if K.image_data_format() == 'channels_first':
        original_img_size = (1, 45, 45)
    else:
        original_img_size = (45, 45, 1)

    EnsIni= sio.loadmat(dirBase)
    x_Facies=np.transpose(EnsIni['Dato']).astype('float32')
    x_train =x_Facies[0:25000]        
    x_test  =x_Facies[25000:29997]
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train = x_train.reshape((x_train.shape[0],) + (original_img_size))
    x_test =  x_test.reshape((x_test.shape[0],) + (original_img_size))
    return x_train,x_test

def kl_normal(z_mean, z_log_var):
    """
    KL divergence between N(0,1) and N(z_mean, exp(z_log_var)) where covariance
    matrix is diagonal.

    Parameters
    ----------
    z_mean : Tensor

    z_log_var : Tensor

    dim : int
        Dimension of tensor
    """
    # Sum over columns, so this now has size (batch_size,)
    kl_per_example = .5 * (K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=1))
    return K.mean(kl_per_example)


def kl_discrete(dist):
    """
    KL divergence between a uniform distribution over num_cat categories and
    dist.

    Parameters
    ----------
    dist : Tensor - shape (None, num_categories)

    num_cat : int
    """
    num_categories = tuple(dist.get_shape().as_list())[1]
    dist_sum = K.sum(dist, axis=1)  # Sum over columns, this now has size (batch_size,)
    dist_neg_entropy = K.sum(dist * K.log(dist + EPSILON), axis=1)
    return np.log(num_categories) + K.mean(dist_neg_entropy - dist_sum)


def sampling_concrete(alpha, out_shape, temperature=0.67):
    """
    Sample from a concrete distribution with parameters alpha.

    Parameters
    ----------
    alpha : Tensor
        Parameters
    """
    uniform = K.random_uniform(shape=(K.shape(alpha)))
    #uniform = K.random_uniform(shape=out_shape)
    gumbel = - K.log(- K.log(uniform + EPSILON) + EPSILON)
    logit = (K.log(alpha + EPSILON) + gumbel) / temperature
    return K.softmax(logit)


def sampling_normal(z_mean, z_log_var, out_shape):
    """
    Sampling from a normal distribution with mean z_mean and variance z_log_var
    """
    #epsilon = K.random_normal(shape=out_shape, mean=0., stddev=1.)
    epsilon = K.random_normal(shape=(K.shape(z_mean)), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
