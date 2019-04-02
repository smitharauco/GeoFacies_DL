import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import time
import h5py
from keras import backend as K
from keras.datasets import mnist
from plotly import tools
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf

EPSILON = 1e-8

def Save_Model(modelo,name):
    def save(model, model_name):
        model_path = "%s.json" % model_name
        weights_path = "%s_weights.hdf5" % model_name
        options = {"file_arch": model_path, 
                   "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])
    save(modelo,name)

def PlotHistory(history,listKeys=[],axis_=[]):
    if len(listKeys)==0:
        listKeys=history.keys()
    leg=[]
    for key in listKeys:
        plt.plot(history[key])
        plt.title('Training')
        plt.xlabel('epoch') 
        leg.append(key)
        print(key,"  : ",history[key][-5:-1])
    plt.legend(leg, loc='upper left')
    if len(axis_)>0:
        plt.axis(axis_)

# Plot History
def plot_history(train, test):
    plt.plot(train, color='red', label='train')
    plt.plot(test, color='g', label='test')
    plt.legend()

def PlotDataAE(X,X_AE,digit_size=28,cmap='jet',Only_Result=True):    
    if Only_Result:
        plt.figure(figsize=(digit_size*0.5,digit_size*0.5))
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

# Plot Images    
def plot_images(imagens, name='Inputs'):
    plt.figure(1,figsize=(20,100))
    plt.title(name)
    for i in range(10):
        plt.title(name)
        plt.subplot(1, 10, i+1)
        plt.axis('off')
        plt.imshow((np.argmax(imagens[i],axis=-1)), cmap='jet')
    plt.show()
    

def LoadMPS45(dirBase='/work/Home89/PythonUtils/DataSetThesis/MPS45.mat',ShortDate=False,AllTrain=False):
    if K.image_data_format() == 'channels_first':
        original_img_size = (1, 45, 45)
    else:
        original_img_size = (45, 45, 1)
    EnsIni= sio.loadmat(dirBase)
    x_Facies=np.transpose(EnsIni['Dato']).astype('float32')
    
    if AllTrain :
        x_train =x_Facies
    else:
        if ShortDate :
            x_train =x_Facies[0:5000]
        else :
            x_train =x_Facies[0:25000]
            
    x_test  =x_Facies[25000:29997]
    x_train = x_train.reshape((x_train.shape[0],) + (original_img_size))
    x_test =  x_test.reshape((x_test.shape[0],) + (original_img_size))
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    return x_train,x_test

def LoadMPS100(dirBase='/work/Home89/PythonUtils/DataSetThesis/MPS100.mat',ShortDate=False,AllTrain=False):
    if K.image_data_format() == 'channels_first':
        original_img_size = (1, 100, 100)
    else:
        original_img_size = (100, 100, 1)
    x_Facies = {}
    f = h5py.File(dirBase)
    for k, v in f.items():
        x_Facies[k] = np.array(v)      
    x_Facies=x_Facies['Dato'].astype('float32')
    f.close()
    if AllTrain :
        x_train =x_Facies
    else :
        if ShortDate :
            x_train =x_Facies[0:5000]
        else :
            x_train =x_Facies[0:32000]
            
    x_test  =x_Facies[32000:40000]
    x_train = x_train.reshape((x_train.shape[0],) + (original_img_size))
    x_test =  x_test.reshape((x_test.shape[0],) + (original_img_size))
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    resultado =tf.Variable(initial,name='w')
    #print(resultado.name)
    return resultado

    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    resultado = tf.Variable(initial, name='b')
    #print(resultado.name)
    return resultado


def Dense(prev, input_size, output_size,reuse=tf.AUTO_REUSE):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(prev, W) + b
