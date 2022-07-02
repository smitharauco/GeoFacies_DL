import numpy as np

import time
import h5py
from keras import backend as K
#import plotly.graph_objs as go
#import plotly.offline as py
#from plotly import tools
# from keras.datasets import mnist
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.data_utils import get_file
import requests
import wget


EPSILON = 1e-8

ID_MPS45_TRAIN = "1SrCyt8Vd9eQOV1b-biFRVimbdOS2-_TF"
ID_MPS45_TEST = "1DNE_HJGt96zoGQDqUMTwgHBZkAC0yxqI"
ID_MPS60_TRAIN = "1s5s8eDOi_sAcarzNsy4WEVRoc9Tiitwp"
ID_MPS60_TEST = "1QzsLTRKBIIzoCtYObfa_EHI2TcDSAJi7"
ID_MPS100_TRAIN = "16eZeCRIx2stRfreBbJshosCLJEnVcZ_c"
ID_MPS100_TEST = "1KrUQUJfGO1wS69tA9gEiZzPJYyiGHzak"
ID_MPS40x200_TRAIN = "1nGaE98hvO8ljDB9pvYQTk817h6Kn6xE9"
ID_MPS40x200_TEST = "1YtZjbeWTFOQHCk-oQ3Rd_ClNG72mUCtR"
    

def MakeDirectory(path):  # mkdir_p
    import os
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

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

def PlotHistory(history,listKeys=[],axis_=[],save=False,save_name=''):
    if len(listKeys)==0:
        listKeys=history.keys()
    leg=[]
    for key in listKeys:
        plt.plot(history[key][5:])
        plt.title('Training')
        plt.xlabel('epoch') 
        leg.append(key)
        print(key,"  : ",history[key][-5:-1])
    plt.legend(leg, loc='upper left')
    if len(axis_)>0:
        plt.axis(axis_)
    if save:
        plt.savefig(save_name)

# Plot History
def plot_history(train, test):
    plt.plot(train, color='red', label='train')
    plt.plot(test, color='g', label='test')
    plt.legend()


def PlotDataAE(X,X_AE,digit_size=(28,28),cmap='jet',Only_Result=True,num=10,figsize=None, path_file = None):  
    if figsize is None:
        figsize=(digit_size[0]*0.5,digit_size[1]*0.5)
    try:
        digit_size=(digit_size[0],digit_size[1])
    except :
        digit_size=(digit_size,digit_size)
            
    if Only_Result:
        plt.figure(figsize=figsize)
        for i in range(0,num):
            plt.subplot(num,num,(i+1))
            plt.imshow(np.squeeze(X[i].reshape(digit_size[0],digit_size[1],1)),cmap=cmap)
            plt.axis('off')
            plt.title('Input')
        plt.show()
        
    plt.figure(figsize=figsize)
    for i in range(0,num):
        plt.subplot(num,num,(i+1))
        plt.imshow(np.squeeze(X_AE[i].reshape(digit_size[0],digit_size[1],1)),cmap=cmap) 
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

def kl_normal(z_mean, z_log_var, weight=.5):
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
    kl_per_example = weight * (K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=1))
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


def load_numpy(path,split_data=0.33,random_state=0,return_to_categorical=True):

    """load dataset from .npy"""

    data = np.load(path)
    if return_to_categorical:
        data = to_categorical(data,2)
    x_train, x_test  = train_test_split(data, test_size=split_data,random_state=random_state)
    return x_train, x_test

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, name, path_download):
    
    """Converts a dataset to tfrecords."""
    
    images = data_set

    num_examples = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(path_download, name + '.tfrecords')
    print('Writing', filename)
    
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            
            image_raw = images[index].astype(np.uint8)
        
            image_raw = image_raw.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'image_raw': _bytes_feature(image_raw),
                'num_samples': _int64_feature(num_examples)
                }))
            writer.write(example.SerializeToString())


def convert_to_tfrecords(path, x_train, x_test_val):
    convert_to(x_train, 'train', path)
    convert_to(x_test_val, 'test_val', path)
    #convert_to(x_test, 'test', path)

def load_from_tfrecords(path, batch_size):
    path_train = os.path.join(path, 'train.tfrecords')
    #path_val = os.path.join(path, 'val.tfrecords')
    path_test = os.path.join(path, 'test_val.tfrecords')
    gen_train = MPS_Generator(path_train, batch_size)
    #gen_val = MPS_Generator(path_val, batch_size)
    gen_test = MPS_Generator(path_test, batch_size)

    return gen_train, gen_test

class MPS_Generator():

    """Class to create a generator to train with tfrecords"""
    
    def __init__(self, path = '', _batch_size = 10):
          
        self.dataset = tf.data.TFRecordDataset(path).map(self.decode_example).repeat().shuffle(42).batch(_batch_size)
        self._batch_size = _batch_size
        
        _iter = self.dataset.make_one_shot_iterator()
        batch = _iter.get_next()
        first_bacth = K.batch_get_value(batch)
        self.image_dim = first_bacth[0].shape[1:]       
        self.num = np.unique(first_bacth[1])[0]

    def decode_example(self, example_proto):

        features = tf.parse_single_example(
            example_proto,
            features = {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([],tf.int64),
                'width': tf.FixedLenFeature([],tf.int64),
                'depth': tf.FixedLenFeature([],tf.int64),
                'num_samples': tf.FixedLenFeature([],tf.int64)
          }
        )

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, (features['height'], features['width'], features['depth']))
        
        num = features['num_samples']

        return [image, num]

    def mps_generator(self):

        _iter = self.dataset.make_one_shot_iterator()
        batch = _iter.get_next()

        while True:
            x = K.batch_get_value(batch)[0]
            yield (x, x)

    def get_numpy_batch(self):

        _iter = self.dataset.make_one_shot_iterator()
        batch = _iter.get_next()
        x = K.batch_get_value(batch)[0]
        return x

    def __len__(self):
        return self.num // self._batch_size


def GetPCAComponents(diagonal,epsilon):
    sum_diagonal    = np.sum(diagonal)
    cumsum_diagonal = np.cumsum(diagonal)
    result = cumsum_diagonal/sum_diagonal
    Nr=result[result < epsilon].shape[0]
    return np.diag(diagonal[0:Nr+1])

def model_plot(data, Model = None, mode = "reconstruction", type_model = "cvae", num_plots = 10, figsize = (None,None)):

    """Test, generate and plot any model"""
    # mode: reconstruction/gen
    # type_model: ('cvae','cvae_style','CycleGAN')

    if mode == "reconstruction":
        _,n_rows,n_cols,_ = data.shape
        data_procs = Model.model.predict(data)
        Only_Result = True
    else:
        n_rows, n_cols = figsize
        if type_model != "CycleGAN":
            data_procs = Model.generator.predict(data)
        else:
            data_procs = Model.decoder.predict(data)
        Only_Result = False
        data = np.zeros((1,n_rows,n_cols,2)) #auxiliar

    if type_model == "cvae":
        data_procs = np.argmax(data_procs,axis=-1)
        PlotDataAE2(data[:,:,:,1], data_procs, digit_size=(n_rows,n_cols), Only_Result=Only_Result, num=num_plots)

    if type_model == "cvae_style":
        data_procs[data_procs > 0] = 1
        data_procs[data_procs <= 0] = 0
        PlotDataAE2(data,data_procs,digit_size=(n_rows,n_cols),Only_Result=Only_Result, num=num_plots)

    if type_model == "CycleGAN":
        data_procs = np.argmax(data_procs,axis=-1)
        PlotDataAE2(np.argmax(data, axis=-1), data_procs, digit_size=(n_rows, n_cols),Only_Result=Only_Result, num=num_plots)


def PlotDataAE2(X,X_AE,digit_size=(28,28),cmap='jet',Only_Result=True,num=10,figsize=None, path_file = None):  
    if figsize is None:
        figsize=(digit_size[0]*0.5,digit_size[1]*0.5)
    try:
        digit_size=(digit_size[0],digit_size[1])
    except :
        digit_size=(digit_size,digit_size)

    if Only_Result:
        fig, axarr = plt.subplots(nrows=2, ncols=num, figsize=(16,4))
        for i in range(0,num):
            axarr[0, i].imshow(np.squeeze(X[i].reshape(digit_size[0],digit_size[1],1)),cmap=cmap)
            axarr[0, i].axis('off')
            axarr[0, i].set_title('Input')
            axarr[1, i].imshow(np.squeeze(X_AE[i].reshape(digit_size[0],digit_size[1],1)),cmap=cmap)
            axarr[1, i].axis('off')
            axarr[1, i].set_title('Output')
    else:
        fig, axarr = plt.subplots(nrows=1, ncols=num, figsize=(16,2))
        for i in range(0,num):
            axarr[i].imshow(np.squeeze(X_AE[i].reshape(digit_size[0],digit_size[1],1)),cmap=cmap)
            axarr[i].axis('off')
            axarr[i].set_title('Output')
    if path_file:
        plt.savefig(path_file)
    plt.show()
    
    
def download_file_with_wget(id, destination):
    url = "https://drive.google.com/u/0/uc?id="+id
    #https://drive.google.com/u/1/uc?id=1DNE_HJGt96zoGQDqUMTwgHBZkAC0yxqI
    print(url)
    wget.download(url,destination )
    
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)



def download_tfrecord(name_train = "train.tfrecords", path = 'DataSet', name_dataset = "MPS45", test = False, name_test ="test_val.tfrecords" ):

    # name_dataset: ('mps45','mps60','mps100','mps40')

    if name_dataset == "MPS45":
        id_train = ID_MPS45_TRAIN
        if test:
            id_test = ID_MPS45_TEST

    if name_dataset == "MPS60":       
        id_train = ID_MPS60_TRAIN
        if test:
            id_test = ID_MPS60_TEST

    if name_dataset == "MPS100":
        id_train = ID_MPS100_TRAIN
        if test:
            id_test = ID_MPS100_TEST

    if name_dataset == "MPS40x200":
        id_train = ID_MPS40x200_TRAIN
        if test:
            id_test = ID_MPS40x200_TEST

    MakeDirectory(os.path.join(path,name_dataset))
    _path_file_train = os.path.join(path,name_dataset,name_train)

    if os.path.exists(_path_file_train) == False:
        print("Downloading data in", _path_file_train)
        #download_file_from_google_drive(id_train, _path_file_train)
        download_file_with_wget(id_train, _path_file_train)

    if test:
        _path_file_test = os.path.join(path,name_dataset,name_test)
        if os.path.exists(_path_file_test) == False:
            print("Downloading data in", _path_file_test)
            #download_file_from_google_drive(id_test, _path_file_test)
            download_file_with_wget(id_test, _path_file_test)
    print("Data Set is OK ")
