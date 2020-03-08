import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage
import pdb
import math
import sys
import time
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge,Activation,Add,AveragePooling2D, Conv2DTranspose
from keras.layers import Conv2D, Deconv2D, Dropout, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.layers import DepthwiseConv2D,Add,AveragePooling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.objectives import binary_crossentropy
from keras.models import model_from_json
from keras.utils import to_categorical
from Model.BiLinearUp import BilinearUpsampling
from Model.Utils import Save_Model, PlotDataAE
from Model.UtilsPCA import ComputePCA

class Utilities(object):    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        loss =  tf.reduce_mean(binary_crossentropy(y_true, y_pred))
        return loss
    
    @staticmethod
    def cross_entropy_loss(labels, logits):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        return loss
    
    @staticmethod
    def ae_mse(y_true, y_pred):
        loss =  100 * tf.nn.l2_loss(y_true - y_pred)
        return loss
    
    @staticmethod
    def accuracy(y_true, y_pred):
        equal = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
        acc = K.sum(equal) / K.cast(tf.size(equal), K.floatx())
        return acc
    
    @staticmethod
    def accuracy_round(y_true, y_pred):
        y_true = (y_true + 1) / 2.0
        y_pred = (y_pred + 1) / 2.0
        equal = K.cast(K.equal(K.round(y_true), K.round(y_pred)), K.floatx())
        acc = K.sum(equal) / K.cast(tf.size(equal), K.floatx())
        return acc
    
    @staticmethod
    def reconstruction_loss(real, recovered):
        loss = tf.reduce_mean(tf.abs(real - recovered))
        return loss
    
    @staticmethod
    def l1_loss(real, recovered):
        loss = tf.reduce_mean(tf.abs(real - recovered))
        return loss
    
    @staticmethod
    def cycle_loss(real, recovered):
        loss = tf.reduce_mean(tf.abs(real - recovered))
        return loss
    
    @staticmethod
    def acc_cpu(y_true, y_pred):
        equal = np.float32(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)))
        return np.sum(equal) / np.size(equal)
    
    @staticmethod    
    def build_pretrained_model(model_file, isTrain=True):
        with open(model_file + '.json', 'r') as f:
            model = model_from_json(f.read(), custom_objects={'BilinearUpsampling': BilinearUpsampling()})
        model.load_weights(model_file + '_weights.hdf5')
        if isTrain is False:
            model.trainable = False
        return model
        
    @staticmethod
    def process_data(images, flip=False, is_test=False):
        img_A, img_B = images[:, :, :1], images[:, :, 1:]
        img_A, img_B = Utilities().preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)
        img_AB = np.concatenate((img_A, img_B), axis=-1)
        return img_AB

    @staticmethod
    def preprocess_A_and_B(img_A, img_B, flip=True, is_test=False):
        n_rows, n_cols = img_A.shape[0], img_A.shape[1]
        fine_size=(n_rows, n_cols)
        load_size=(int(1.1*n_rows), int(1.1*n_cols))
        if is_test:
            img_A = skimage.transform.resize(img_A, [fine_size[0], fine_size[1]], preserve_range=True)
            img_B = skimage.transform.resize(img_B, [fine_size[0], fine_size[1]], preserve_range=True)
        else:
            img_A = skimage.transform.resize(img_A, [load_size[0], load_size[1]], preserve_range=True)
            img_B = skimage.transform.resize(img_B, [load_size[0], load_size[1]], preserve_range=True)

            h1 = int(np.ceil(np.random.uniform(1e-2, load_size[0]-fine_size[0])))
            w1 = int(np.ceil(np.random.uniform(1e-2, load_size[1]-fine_size[1])))
            img_A = img_A[h1:h1+fine_size[0], w1:w1+fine_size[1]]
            img_B = img_B[h1:h1+fine_size[0], w1:w1+fine_size[1]]

            if flip and np.random.random() > 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

        return img_A, img_B
    
    @staticmethod
    def split_data(data, Nr):
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        trn_data = data[idx[:Nr]]
        val_data = data[idx[Nr:]]
        if np.ndim(data) != 4:
            trn_data = np.expand_dims(trn_data, axis=-1)
            val_data = np.expand_dims(val_data, axis=-1)            
        return trn_data, val_data
    
    @staticmethod
    def random_pca_realizations(dims, pca_model, Nt, plots=True):
        n_rows, n_cols = dims[0], dims[1]
        Us, Us_inv, const, nc, mu = pca_model[0], pca_model[1], pca_model[2], pca_model[3], pca_model[4] 
        z_sample = np.random.normal(0, 1.0, [nc, Nt])
        model_pca = const * np.dot(Us, z_sample) + mu.reshape(n_rows*n_cols, 1)
        model_pca = model_pca.T
        model_pca = model_pca.reshape(-1, n_rows, n_cols, 1)
        if plots:
            PlotDataAE([], model_pca, digit_size=(n_rows, n_cols), Only_Result=False, num=5)
        return model_pca
    
    @staticmethod
    def map2pca(data, dims, pca_model, plots=True):
        n_rows, n_cols = dims[0], dims[1]
        Us, Us_inv, const, _, mu = pca_model[0], pca_model[1], pca_model[2], pca_model[3], pca_model[4] 
        data = data.reshape(-1, n_rows * n_cols)
        # centering data
        data_C = data - mu
        # Encoding
        z_codes = (1.0 / const) * np.dot(Us_inv, data_C.T)
        # Decoding
        data_decoded = (const) * np.dot(Us, z_codes).T + mu
        data_decoded = data_decoded.reshape(-1, n_rows, n_cols, 1)
        if plots:
            PlotDataAE(data, data_decoded, digit_size=(n_rows, n_cols), Only_Result=True, num=5)
        return data_decoded
    
    @staticmethod
    def min_max_samples(data):
        for i in range(len(data)):
            data[i] = 2 * (data[i] - data[i].min()) / (data[i].max() - data[i].min()) - 1.0
        return data