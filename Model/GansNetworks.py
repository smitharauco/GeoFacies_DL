import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage
import pdb
import math
import sys
import time
import keras
from keras.layers import Input, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, AveragePooling2D
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, AveragePooling3D
from keras.layers import Concatenate, Add, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.initializers import RandomNormal
from keras import backend as K
from keras.objectives import binary_crossentropy
from keras.models import model_from_json
from keras.utils import to_categorical
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from Model.BiLinearUp import BilinearUpsampling
from Model.Utils import Save_Model, PlotDataAE
from Model.GansUtilities import Utilities

import random

class Layers2D(object):
    def __init__(self):
        pass
    
    def Conv2D_Block(self,
                     input_x,
                     n_kernels,
                     k_size=4,
                     strides=2,
                     padding='same',
                     kernel_initializer=RandomNormal(stddev=0.02),
                     name='None',
                     activation='leaky_relu',
                     insnorm=False,
                     bn=True,
                     bias=True,
                     bn_training=False):

        x = Conv2D(n_kernels,
                   k_size,
                   strides=strides,
                   padding=padding,
                   kernel_initializer=kernel_initializer,
                   name=name + '_conv2D',
                   use_bias=bias)(input_x)

        if insnorm:
            x = InstanceNormalization(axis=-1, name=name+'_insnorm')(x)
        elif bn:
            x = BatchNormalization(momentum=0.8, name=name + '_bn')(x, training=bn_training)
        if activation is 'leaky_relu':
            x = LeakyReLU(0.2, name=name + '_' + activation)(x)
        else:
            x = Activation(activation, name=name + '_' + activation)(x)
        return x

    def Conv2DTranspose_Block(self,
                              input_x,
                              n_kernels,
                              k_size=4,
                              strides=2,
                              padding='same',                           
                              activation='relu',
                              kernel_initializer=RandomNormal(stddev=0.02),
                              insnorm=False,
                              bn=True,
                              bias=True,
                              bn_training=True,
                              name='None'):

        x = Conv2DTranspose(n_kernels,
                            k_size,
                            strides=strides,
                            padding=padding,
                            kernel_initializer=kernel_initializer,
                            name=name+'_deconv2D',
                            use_bias=bias)(input_x)
        if insnorm:
            x = InstanceNormalization(axis=-1, name=name+'_insnorm')(x)
        elif bn:
            x = BatchNormalization(momentum=0.8, name=name+'_bn')(x, training=bn_training)
        if activation is not 'linear':
            x = Activation(activation, name=name+'_'+activation)(x)
        return x
    
    def residual_block(self, input_x, n_kernels, k_size=4, activation='leaky_relu', insnorm=False, bn_training=False, add=True, name='name'):
        x = self.Conv2D_Block(input_x, n_kernels, k_size=k_size, strides=1, insnorm=insnorm, bn_training=bn_training, activation=activation, name=name+'rba')
        x = self.Conv2D_Block(x, n_kernels, k_size=k_size, strides=1, insnorm=insnorm, bn_training=bn_training, activation='linear', name=name+'rbb')
        if add:
            x = Add(name=name+'add')([x, input_x])
        else:
            x = Concatenate(name=name+'concatenate')([x, input_x])
        return x

class Networks(object):
    def __init__(self):
        self.utils = Utilities()
        self.Conv2D_Block = Layers2D().Conv2D_Block
        self.Conv2DTranspose_Block = Layers2D().Conv2DTranspose_Block
        self.residual_block = Layers2D().residual_block
        
    def build_discriminator2D(self, model_shape, filters=32, k_size=4, drop=True, rate=0.5, summary=False, model_file=None, name='gan_d_'):
        """
        Create a Discriminator Model using hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            """
            Create a Discriminator Model using hyperparameters values defined as follows
            """
            n_rows = model_shape[0]
            n_cols = model_shape[1]
            c_dims = model_shape[2]

            input_shape  = (n_rows, n_cols, c_dims)            
            input_layer  = Input(shape=input_shape, name=name+'input')

            d = self.Conv2D_Block(input_layer, filters, k_size=k_size, name=name+'1', bn=False)    # 30x30x32
            d = self.Conv2D_Block(d, 2*filters, k_size=k_size, name=name+'2')  # 15x15x64
            d = self.Conv2D_Block(d, 4*filters, k_size=k_size, name=name+'3')  # 8x8x128
            d = self.Conv2D_Block(d, 8*filters, strides=1, k_size=k_size, name=name+'4')  # 8x8x256

            d = Flatten(name=name+'flatten')(d)
            if drop:
                d = Dropout(rate=rate, name=name+'dropout')(d, training=True)
            logits = Dense(1, activation='linear', kernel_initializer=RandomNormal(stddev=0.02), name=name+'dense')(d)
            out = Activation('sigmoid', name=name+'sigmoid')(logits)

            model = Model(inputs=[input_layer], outputs=[out, logits], name='Discriminator')
            if (summary):
                model.summary()
            return model
        
    def build_generator2D(self, model_shape, filters=32, k_size=4, z_size=500, summary=False, model_file=None, name='gan_g_'):
        """
        Create a Generator Model with hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:

            n_rows = model_shape[0]
            n_cols = model_shape[1]
            input_shape = (z_size,)
            if n_rows % 8 !=0:
                height = n_rows//8 + 1
            else:
                height = n_rows//8
            if n_cols % 8 !=0:
                width = n_cols//8 +1
            else:
                width = n_cols//8
                            
            num_init_neurons = 8*filters 
            reshape_size= (height, width, num_init_neurons)
            
            # 8*height, 4*height, 2*height, height = n_rows, n_rows//2, n_rows//4, n_rows//8
            rows_matching = np.equal([2*height, 4*height, 8*height], [n_rows//4, n_rows//2, n_rows])
            index_rows = np.where(np.logical_not(rows_matching))[0]
            if len(index_rows) > 0:
                 index_rows = index_rows[0]
            # print(index_rows)
            # 8*width, 4*width, 2*width, width = n_cols//1, n_cols//2, n_cols//4, n_cols//8
            cols_matching = np.equal([2*width, 4*width, 8*width], [n_cols//4, n_cols//2, n_cols])
            index_cols = np.where(np.logical_not(cols_matching))[0]
            if len (index_cols) > 0:
                 index_cols = index_cols[0]
            # print(index_cols)

            input_layer = Input(shape=input_shape, name=name+'input')
            g = Dense(width * height * num_init_neurons, kernel_initializer=RandomNormal(stddev=0.02), name=name+'dense')(input_layer)
            g = Reshape(reshape_size, name=name+'reshape')(g)
            g = BatchNormalization(momentum=0.8, name=name+'bn_dense')(g, training=True)
            g = Activation(activation='relu', name=name+'relu')(g)

            g = self.Conv2DTranspose_Block(g, 4*filters, name=name+'1')
            if index_rows==0 or index_cols==0:
                g = BilinearUpsampling(output_size=(n_rows//4, n_cols//4), name=name+'bilinear')(g)
            g = self.Conv2DTranspose_Block(g, 2*filters, k_size=k_size, name=name+'2')
            if index_rows==1 or index_cols==1:
                g = BilinearUpsampling(output_size=(n_rows//2, n_cols//2), name=name+'bilinear')(g)
            g = self.Conv2DTranspose_Block(g, 1*filters, k_size=k_size, name=name+'3')
            if index_rows==2 or index_cols==2:
                g = BilinearUpsampling(output_size=(n_rows, n_cols), name=name+'bilinear')(g)            
            g = self.Conv2DTranspose_Block(g, 2, strides=1, activation='tanh', k_size=k_size, name=name+'4', bn=False)

            model = Model(inputs=[input_layer], outputs=[g], name='Generator')
            if (summary):
                model.summary()
            return model

    def build_generator2D_(self, model_shape, filters=32, k_size=4, z_size=500, summary=False, model_file=None, name='gan_g_'):
        """
        Create a Generator Model with hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:

            n_rows = model_shape[0]
            n_cols = model_shape[1]
            c_dims = model_shape[2]
            input_shape = (z_size,)
            if n_rows % 8 !=0:
                height = n_rows//8 + 1
            else:
                height = n_rows//8
            if n_cols % 8 !=0:
                width = n_cols//8 +1
            else:
                width = n_cols//8
                            
            num_init_neurons = 8*filters 
            reshape_size= (height, width, num_init_neurons)
            
            # 8*height, 4*height, 2*height, height = n_rows, n_rows//2, n_rows//4, n_rows//8
            rows_matching = np.equal([2*height, 4*height, 8*height], [n_rows//4, n_rows//2, n_rows])
            index_rows = np.where(np.logical_not(rows_matching))[0]
            if len(index_rows) > 0:
                 index_rows = index_rows[0]
            # print(index_rows)
            # 8*width, 4*width, 2*width, width = n_cols//1, n_cols//2, n_cols//4, n_cols//8
            cols_matching = np.equal([2*width, 4*width, 8*width], [n_cols//4, n_cols//2, n_cols])
            index_cols = np.where(np.logical_not(cols_matching))[0]
            if len (index_cols) > 0:
                 index_cols = index_cols[0]
            # print(index_cols)

            input_layer = Input(shape=input_shape, name=name+'input')
            g = Dense(width * height * num_init_neurons, kernel_initializer=RandomNormal(stddev=0.02), name=name+'dense')(input_layer)
            g = Reshape(reshape_size, name=name+'reshape')(g)
            g = BatchNormalization(momentum=0.8, name=name+'bn_dense')(g, training=True)
            g = Activation(activation='relu', name=name+'relu')(g)

            g = self.Conv2DTranspose_Block(g, 4*filters, name=name+'1')
            if index_rows==0 or index_cols==0:
                g = BilinearUpsampling(output_size=(n_rows//4, n_cols//4), name=name+'bilinear')(g)
            g = self.Conv2DTranspose_Block(g, 2*filters, k_size=k_size, name=name+'2')
            if index_rows==1 or index_cols==1:
                g = BilinearUpsampling(output_size=(n_rows//2, n_cols//2), name=name+'bilinear')(g)
            g = self.Conv2DTranspose_Block(g, 1*filters, k_size=k_size, name=name+'3')
            if index_rows==2 or index_cols==2:
                g = BilinearUpsampling(output_size=(n_rows, n_cols), name=name+'bilinear')(g)            
            g = self.Conv2DTranspose_Block(g, c_dims, strides=1, activation='tanh', k_size=k_size, name=name+'4', bn=False)

            model = Model(inputs=[input_layer], outputs=[g], name='Generator')
            if (summary):
                model.summary()
            return model
        
    def build_encoder2D(self, model_shape, filters=32, k_size=4, z_size=500, drop=True, rate=0.5, bn=True, summary=False, model_file=None, name='gan_e_'):
        """
        Create a Discriminator Model using hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            """
            Create a Discriminator Model using hyperparameters values defined as follows
            """
            n_rows = model_shape[0]
            n_cols = model_shape[1]
            c_dims = model_shape[2]

            input_shape = (n_rows, n_cols, c_dims)
            input_layer = Input(shape=input_shape, name=name+'input')

            x = self.Conv2D_Block(input_layer, filters, k_size=k_size, name=name+'1', bn=False)
            x = self.Conv2D_Block(x, 2*filters, k_size=k_size, name=name+'2')
            x = self.Conv2D_Block(x, 4*filters, k_size=k_size, name=name+'3')
            x = self.Conv2D_Block(x, 8*filters, strides=1, k_size=k_size, name=name+'4')

            x = Flatten(name=name+'flatten')(x)
            if drop:
                x = Dropout(rate=rate, name=name+'dropout')(x)
            x = Dense(z_size, activation='linear', kernel_initializer=RandomNormal(stddev=0.02), name=name+'dense')(x)
            if bn:
                x = BatchNormalization(center=False, scale=False, name=name+"bn_out")(x)

            model = Model(inputs=[input_layer], outputs=[x], name='Encoder')
            if (summary):
                model.summary()
            return model

    def build_code_discriminator(self, filters=3500, z_size=500, summary=False,  model_file=None, name='gan_c_'):
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            
            init = RandomNormal(stddev=0.02)
            input_shape  = (z_size,)
            input_layer = Input(shape=input_shape, name=name+'input')

            a = Dense(filters, kernel_initializer=init, name=name+'dense_1')(input_layer)
            a = LeakyReLU(0.2, name=name+'act_1')(a)

            a = Dense(filters, kernel_initializer=init, name=name+'dense_2')(a)
            a = LeakyReLU(0.2, name=name+'act_2')(a)

            logits = Dense(1, kernel_initializer=init, name=name+'dense_3')(a)
            out = Activation('sigmoid', name=name+'act')(logits)

            model = Model(inputs=[input_layer], outputs=[out, logits], name='code_discriminator')
            if (summary):
                model.summary()
            return model

    def build_patch_discriminator(self, model_shape, filters=32, k_size=4, drop=False, rate=0.5, summary=False, model_file=None, name='gan_d_'):
        """
        Create a Discriminator Model using hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            """
            Create a Discriminator Model using hyperparameters values defined as follows
            """
            init = RandomNormal(stddev=0.02)
            n_rows = model_shape[0]
            n_cols = model_shape[1]
            c_dims = model_shape[2]

            input_shape  = (n_rows, n_cols, c_dims)
            input_layer  = Input(shape=input_shape, name=name+'input')

            d = self.Conv2D_Block(input_layer, filters, k_size=k_size, name=name+'1', bn=False)
            d = self.Conv2D_Block(d, 2*filters, k_size=k_size, name=name+'2')
            d = self.Conv2D_Block(d, 4*filters, k_size=k_size, name=name+'3')
            d = self.Conv2D_Block(d, 8*filters, strides=1, k_size=k_size, name=name+'4')
            d = self.Conv2D_Block(d, 8*filters, strides=1, k_size=k_size, name=name+'5')

            if drop:
                d = Dropout(rate=0.5, name=name+'_dropout')(d, training=True)
            logits = Conv2D(1, k_size, strides=1, padding='same', kernel_initializer=init, name=name+'logits')(d)
            out = Activation('sigmoid', name=name+'sigmoid')(logits)

            model = Model(inputs=[input_layer], outputs=[out, logits], name='Discriminator_'+name[-3:])
            if (summary):
                model.summary()
            return model
    
    def build_patch_discriminator_(self, model_shape, filters=32, k_size=4, drop=False, rate=0.5, summary=False, model_file=None, name='gan_d_'):
        """
        Create a Discriminator Model using hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            """
            Create a Discriminator Model using hyperparameters values defined as follows
            """
            init = RandomNormal(stddev=0.02)
            n_rows = model_shape[0]
            n_cols = model_shape[1]
            c_dims = model_shape[2]

            input_shape  = (n_rows, n_cols, c_dims)            
            input_layer  = Input(shape=input_shape, name=name+'input')

            d = self.Conv2D_Block(input_layer, filters, k_size=k_size, name=name+'1', bn=False)
            d = self.Conv2D_Block(d, 2*filters, k_size=k_size, insnorm=True, name=name+'2')
            d = self.Conv2D_Block(d, 4*filters, k_size=k_size, insnorm=True, name=name+'3')
            d = self.Conv2D_Block(d, 8*filters, k_size=k_size, insnorm=True, strides=1, name=name+'4')

            if drop:
                d = Dropout(rate=0.5, name=name+'_dropout')(d, training=True)
            logits = Conv2D(1, k_size, strides=1, padding='same', kernel_initializer=init, name=name+'logits')(d)
            out = Activation('sigmoid', name=name+'sigmoid')(logits)

            model = Model(inputs=[input_layer], outputs=[out, logits], name='Discriminator_'+name[-3:])
            if (summary):
                model.summary()
            return model
        
    def build_resnet_generator(self, model_shape, filters=32, k_size=3, last_act='tanh', summary=False, model_file=None, name='gan_g_'):
        """
        Create a Generator Model with hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            init = RandomNormal(stddev=0.02)
            n_rows = model_shape[0]
            n_cols = model_shape[1]
            in_c_dims = model_shape[2]
            out_c_dims = model_shape[3]
            
            n_rows_e1, n_rows_e2, n_rows_e4, n_rows_e8 = n_rows//1, n_rows//2, n_rows//4, n_rows//8
            rows_matching = np.equal([2*n_rows_e2, 2*n_rows_e4, 2*n_rows_e8], [n_rows_e1, n_rows_e2, n_rows_e4])
            index_rows = np.where(np.logical_not(rows_matching))[0]
            
            n_cols_e1, n_cols_e2, n_cols_e4, n_cols_e8 = n_cols//1, n_cols//2, n_cols//4, n_cols//8
            cols_matching = np.equal([2*n_cols_e2, 2*n_cols_e4, 2*n_cols_e8], [n_cols_e1, n_cols_e2, n_cols_e4])
            index_cols = np.where(np.logical_not(cols_matching))[0]
          
            input_shape = (n_rows, n_cols, in_c_dims)
            input_layer = Input(shape=input_shape, name=name+'_input')
            
            e1 = self.Conv2D_Block(input_layer, n_kernels=filters, k_size=7, strides=1, bn=False,name=name+'e1') # rows, cols
            e2 = self.Conv2D_Block(e1, 2*filters, k_size=k_size, bn_training=True, name=name+'e2') # rows/2, cols/2
            e3 = self.Conv2D_Block(e2, 4*filters, k_size=k_size, bn_training=True, name=name+'e3') # rows/4, cols/4
            e4 = self.Conv2D_Block(e3, 8*filters, k_size=k_size, bn=False, name=name+'e4') # rows/8, cols/8

            rb1 = self.residual_block(e4,  n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'1_')
            rb2 = self.residual_block(rb1, n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'2_')
            rb3 = self.residual_block(rb2, n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'3_')
            rb3 = Dropout(rate=0.5, name=name+'drop_1')(rb3, training=True)
            
            rb4 = self.residual_block(rb3, n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'4_')
            rb4 = Dropout(rate=0.5, name=name+'drop_2')(rb4, training=True)    
            
            rb5 = self.residual_block(rb4, n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'5_')
            rb5 = Dropout(rate=0.5, name=name+'drop_3')(rb5, training=True)   
            
            d1 = self.Conv2DTranspose_Block(rb5, 4*filters, k_size=k_size, activation='linear', name=name+'d1') # rows/4, cols/4
            if index_rows==2 or index_cols==2:
                d1 = BilinearUpsampling(output_size=(n_rows//4, n_cols//4), name=name+'_bilinear')(d1)
            d1 = Concatenate(name=name+'conc_1')([d1, e3])
            d1 = Activation('relu', name=name+'_act_1')(d1)
            
            d2 = self.Conv2DTranspose_Block(d1, 2*filters, k_size=k_size, activation='linear', name=name+'d2') # rows/2, cols/2
            if index_rows==1 or index_cols==1:
                d2 = BilinearUpsampling(output_size=(n_rows//2, n_cols//2), name=name+'_bilinear')(d2)
            d2 = Concatenate(name=name+'conc_2')([d2, e2])
            d2 = Activation('relu', name=name+'_act_2')(d2)
            
            d3 = self.Conv2DTranspose_Block(d2, 1*filters, k_size=k_size, activation='linear', name=name+'d3') # rows, cols
            if index_rows==0 or index_cols==0:
                d3 = BilinearUpsampling(output_size=(n_rows, n_cols), name=name+'_bilinear')(d2)
            d3 = Concatenate(name=name+'conc_3')([d3, e1])
            d3 = Activation('relu', name=name+'act_3')(d3)

            output = Conv2DTranspose(out_c_dims, 7, strides=1, padding='same',  kernel_initializer=init, name=name+'d_out')(d3) # rows, cols
            # output = InstanceNormalization(axis=-1, name=name+'ins_norm')(output)
            output = Activation(last_act, name=name+last_act)(output)

            model = Model(inputs=[input_layer], outputs=[output], name='Generator'+name[-3:])
            if (summary):
                model.summary()
            return model
    
    def build_resnet_generator_insnorm(self, model_shape, filters=32, k_size=3, last_act='tanh', n_residuals=9, summary=False, model_file=None, name='gan_g_'):
        """
        Create a Generator Model with hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            init = RandomNormal(stddev=0.02)
            n_rows = model_shape[0]
            n_cols = model_shape[1]
            in_c_dims = model_shape[2]
            out_c_dims = model_shape[3]
            
            n_rows_e1, n_rows_e2, n_rows_e4, n_rows_e8 = n_rows//1, n_rows//2, n_rows//4, n_rows//8
            rows_matching = np.equal([2*n_rows_e2, 2*n_rows_e4, 2*n_rows_e8], [n_rows_e1, n_rows_e2, n_rows_e4])
            index_rows = np.where(np.logical_not(rows_matching))[0]
            
            n_cols_e1, n_cols_e2, n_cols_e4, n_cols_e8 = n_cols//1, n_cols//2, n_cols//4, n_cols//8
            cols_matching = np.equal([2*n_cols_e2, 2*n_cols_e4, 2*n_cols_e8], [n_cols_e1, n_cols_e2, n_cols_e4])
            index_cols = np.where(np.logical_not(cols_matching))[0]
          
            input_shape = (n_rows, n_cols, in_c_dims)
            input_layer = Input(shape=input_shape, name=name+'_input')
            
            x = self.Conv2D_Block(input_layer, filters, k_size=7, strides=1, activation='relu', bn=False, name=name+'e1') # rows, cols
            x = self.Conv2D_Block(x, 2*filters, k_size=k_size, insnorm=True, activation='relu', name=name+'e2') # rows/2, cols/2
            x = self.Conv2D_Block(x, 4*filters, k_size=k_size, insnorm=True, activation='relu', name=name+'e3') # rows/4, cols/4
            
            for i in range(n_residuals):
                x = self.residual_block(x, n_kernels=4*filters, k_size=k_size, activation='relu', insnorm=True, add=False, name=name+str(i+1)+'_')
                
            x = self.Conv2DTranspose_Block(x, 2*filters, k_size=k_size, insnorm=True, name=name+'d1') # rows/2, cols/2            
            x = self.Conv2DTranspose_Block(x, 1*filters, k_size=k_size, insnorm=True, name=name+'d2') # rows, cols

            x = Conv2DTranspose(out_c_dims, 7, strides=1, padding='same',  kernel_initializer=init, name=name+'d_out')(x) # rows, cols
            x = InstanceNormalization(axis=-1, name=name+'ins_norm')(x)
            output = Activation(last_act, name=name+last_act)(x)

            model = Model(inputs=[input_layer], outputs=[output], name='Generator'+name[-3:])
            if (summary):
                model.summary()
            return model
        
class Layers3D(object):
    def __init__(self):
        pass
    
    def Conv3D_Block(self,
                     input_x,
                     n_kernels,
                     k_size=4,
                     strides=(2, 2, 2),
                     padding='same',
                     activation='leaky_relu',
                     kernel_initializer=RandomNormal(stddev=0.02),                     
                     insnorm=False,
                     bn=True,
                     bias=True,
                     bn_training=False,
                     name='None'):

        x = Conv3D(n_kernels,
                   k_size,
                   strides=strides,
                   padding=padding,
                   kernel_initializer=kernel_initializer,
                   name=name + '_conv3D',
                   use_bias=bias)(input_x)

        if insnorm:
            x = InstanceNormalization(axis=-1, name=name+'_insnorm')(x)
        elif bn:
            x = BatchNormalization(momentum=0.8, name=name + '_bn')(x, training=bn_training)
        if activation is 'leaky_relu':
            x = LeakyReLU(0.2, name=name + '_' + activation)(x)
        else:
            x = Activation(activation, name=name + '_' + activation)(x)
        return x

    def Conv3DTranspose_Block(self,
                              input_x,
                              n_kernels,
                              k_size=4,
                              strides=(2, 2, 2),
                              padding='same',                           
                              activation='relu',
                              kernel_initializer=RandomNormal(stddev=0.02),
                              insnorm=False,
                              bn=True,
                              bias=True,
                              bn_training=True,
                              name='None'):

        x = Conv3DTranspose(n_kernels,
                            k_size,
                            strides=strides,
                            padding=padding,
                            kernel_initializer=kernel_initializer,
                            name=name+'_deconv2D',
                            use_bias=bias)(input_x)
        if insnorm:
            x = InstanceNormalization(axis=-1, name=name+'_insnorm')(x)
        elif bn:
            x = BatchNormalization(momentum=0.8, name=name+'_bn')(x, training=bn_training)
        if activation is not 'linear':
            x = Activation(activation, name=name+'_'+activation)(x)
        return x
    
    def Residual3D_Block(self, input_x, n_kernels, k_size=4, activation='leaky_relu', insnorm=False, bn_training=False, add=True, name='name'):
        x = self.Conv3D_Block(input_x, n_kernels, k_size=k_size, strides=1, insnorm=insnorm, bn_training=bn_training, activation=activation, name=name+'rba')
        x = self.Conv3D_Block(x, n_kernels, k_size=k_size, strides=1, insnorm=insnorm, bn_training=bn_training, activation='linear', name=name+'rbb')
        if add:
            x = Add(name=name+'add')([x, input_x])
        else:
            x = Concatenate(name=name+'concatenate')([x, input_x])
        return x
    
class Networks3D(object):
    def __init__(self):
        self.Conv3D_Block = Layers2D().Conv2D_Block
        self.Conv3DTranspose_Block = Layers2D().Conv2DTranspose_Block
        self.Residual3D_Block = Layers2D().Residual3D_Block
    
    def build_patch_discriminator3D(self, model_shape, filters=32, k_size=4, drop=True, rate=0.5, summary=False, model_file=None, name='gan_d_'):
        """
        Create a Discriminator Model using hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            """
            Create a Discriminator Model using hyperparameters values defined as follows
            """
            init = RandomNormal(stddev=0.02)
            n_rows = model_shape[0]
            n_cols = model_shape[1]
            n_layers = model_shape[2]
            c_dims = model_shape[3]

            input_shape  = (n_rows, n_cols, n_layers, c_dims)            
            input_layer  = Input(shape=input_shape, name=name+'input')

            d = self.Conv3D_Block(input_layer, filters, name=name+'1', bn=False)
            d = self.Conv3D_Block(d, 2*filters, name=name+'2')
            d = self.Conv3D_Block(d, 4*filters, name=name+'3')
            d = self.Conv3D_Block(d, 8*filters, strides=1, name=name+'4')

            feat = Flatten(name=name+'flatten')(d)
            if drop:
                d = Dropout(rate=0.5, name=name+'_dropout')(d, training=True)
            logits = Conv3D(1, 4, strides=1, padding='same', kernel_initializer=init, name=name+'logits')(d)
            out = Activation('sigmoid', name=name+'sigmoid')(logits)

            model = Model(inputs=[input_layer], outputs=[out, logits, feat], name='Discriminator_'+name[-3:])
            if (summary):
                model.summary()
            return model
    
    def build_resnet_generator3D(self, model_shape, filters=32, k_size=3, last_act='tanh', summary=False, model_file=None, name='gan_g_'):
        """
        Create a Generator Model with hyperparameters values defined as follows
        """
        if (model_file):
            """
            Load pretreined model
            """
            model = self.utils.build_pretrained_model(model_file)
            if (summary):
                model.summary()
            return model
        else:
            init = RandomNormal(stddev=0.02)
            n_rows = model_shape[0]
            n_cols = model_shape[1]
            n_layers = model_shape[2]
            in_c_dims = model_shape[3]
            out_c_dims = model_shape[4]
            
            n_rows_e1, n_rows_e2, n_rows_e4, n_rows_e8 = n_rows//1, n_rows//2, n_rows//4, n_rows//8
            rows_matching = np.equal([2*n_rows_e2, 2*n_rows_e4, 2*n_rows_e8], [n_rows_e1, n_rows_e2, n_rows_e4])
            index_rows = np.where(np.logical_not(rows_matching))[0]
            
            n_cols_e1, n_cols_e2, n_cols_e4, n_cols_e8 = n_cols//1, n_cols//2, n_cols//4, n_cols//8
            cols_matching = np.equal([2*n_cols_e2, 2*n_cols_e4, 2*n_cols_e8], [n_cols_e1, n_cols_e2, n_cols_e4])
            index_cols = np.where(np.logical_not(cols_matching))[0]
          
            input_shape = (n_rows, n_cols, in_c_dims)
            input_layer = Input(shape=input_shape, name=name+'_input')
            
            e1 = self.Conv3D_Block(input_layer, n_kernels=filters, k_size=7, strides=1, bn=False,name=name+'e1') # rows, cols
            e2 = self.Conv3D_Block(e1, 2*filters, k_size=k_size, bn_training=True, name=name+'e2') # rows/2, cols/2
            e3 = self.Conv3D_Block(e2, 4*filters, k_size=k_size, bn_training=True, name=name+'e3') # rows/4, cols/4
            e4 = self.Conv3D_Block(e3, 8*filters, k_size=k_size, bn=False, name=name+'e4') # rows/8, cols/8

            rb1 = self.Residual3D_Block(e4,  n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'1_')
            rb2 = self.Residual3D_Block(rb1, n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'2_')
            rb3 = self.Residual3D_Block(rb2, n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'3_')
            rb3 = Dropout(rate=0.5, name=name+'drop_1')(rb3, training=True)
            
            rb4 = self.Residual3D_Block(rb3, n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'4_')
            rb4 = Dropout(rate=0.5, name=name+'drop_2')(rb4, training=True)    
            
            rb5 = self.Residual3D_Block(rb4, n_kernels=8*filters, k_size=k_size, bn_training=True, name=name+'5_')
            rb5 = Dropout(rate=0.5, name=name+'drop_3')(rb5, training=True)   
            
            d1 = self.Conv3DTranspose_Block(rb5, 4*filters, k_size=k_size, activation='linear', name=name+'d1') # rows/4, cols/4
            if index_rows==2 or index_cols==2:
                d1 = BilinearUpsampling(output_size=(n_rows//4, n_cols//4), name=name+'_bilinear')(d1)
            d1 = Concatenate(name=name+'conc_1')([d1, e3])
            d1 = Activation('relu', name=name+'_act_1')(d1)
            
            d2 = self.Conv3DTranspose_Block(d1, 2*filters, k_size=k_size, activation='linear', name=name+'d2') # rows/2, cols/2
            if index_rows==1 or index_cols==1:
                d2 = BilinearUpsampling(output_size=(n_rows//2, n_cols//2), name=name+'_bilinear')(d2)
            d2 = Concatenate(name=name+'conc_2')([d2, e2])
            d2 = Activation('relu', name=name+'_act_2')(d2)
            
            d3 = self.Conv3DTranspose_Block(d2, 1*filters, k_size=k_size, activation='linear', name=name+'d3') # rows, cols
            if index_rows==0 or index_cols==0:
                d3 = BilinearUpsampling(output_size=(n_rows, n_cols), name=name+'_bilinear')(d2)
            d3 = Concatenate(name=name+'conc_3')([d3, e1])
            d3 = Activation('relu', name=name+'act_3')(d3)

            output = Conv3DTranspose(out_c_dims, 7, strides=1, padding='same',  kernel_initializer=init, name=name+'d_out')(d3) # rows, cols
            output = Activation(last_act, name=name+last_act)(output)

            model = Model(inputs=[input_layer], outputs=[output], name='Generator'+name[-3:])
            if (summary):
                model.summary()
            return model