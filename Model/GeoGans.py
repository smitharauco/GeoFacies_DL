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
from Model.GansUtilities import Utilities
from Model.GansNetworks import Networks

import random
#from UtilsPCA import ComputePCA
from Model.UtilsPCA import ComputePCA

class Printer():
    """Print things to stdout on one line dynamically"""
    def __init__(self, data):
        sys.stdout.write("\r\x1b[K"+data.__str__())
        sys.stdout.flush()
""" ////////////////////////////////  2D CLASSES //////////////////////////////////////////// """
class GAN2D_MPS(object):
    def __init__(self, input_shape,
                d_filters=32, g_filters=32, e_filters=32,
                d_ksize=4, g_ksize=4, e_ksize=4, z_size=500, batch_size=64, d_drop=False,
                d_lr=0.0002, g_lr=0.0002, e_lr=0.001, z_lr=0.1, beta1=0.5,
                model_file=None, saving_path=None, name='geo_GANs_', summary=True
                ):
        if saving_path is None:
            saving_path = './'
        self.saving_path = saving_path
        self.n_rows = input_shape[0]
        self.n_cols = input_shape[1]
        self.c_dims = input_shape[2]
        self.name_base = name+'z_'+str(z_size)+'_'+str(self.n_rows)+'x'+str(self.n_cols)
        self.discriminator_name = self.name_base + '_discriminator'
        self.decoder_name = self.name_base + '_decoder'
        self.encoder_name = self.name_base + '_encoder'
        self.batch_size = batch_size        
        self.d_filters = d_filters
        self.g_filters = g_filters
        self.e_filters = e_filters
        self.d_ksize = d_ksize
        self.g_ksize = g_ksize
        self.e_ksize = e_ksize
        self.z_size = z_size
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.e_lr = e_lr
        self.z_lr = z_lr
        self.d_drop = d_drop
        self.beta1 = beta1
        self.utils = Utilities()
        self.nets = Networks()        
        self.model_file = model_file
        self.summary = summary
        self.build_model(self.model_file, summary=self.summary)
    
    def build_model(self, model_file=None, model_file_enc=None, summary=False):
        
        if model_file is not None:
            model_file_dis=model_file+'_discriminator'
            model_file_gen=model_file+'_decoder'
        else:
            model_file_dis=model_file
            model_file_gen=model_file

        # build network models
        K.clear_session()
        self.decoder = self.nets.build_generator2D(model_shape=(self.n_rows, self.n_cols, self.c_dims), z_size=self.z_size,
                                                   filters=self.g_filters, k_size=self.g_ksize, model_file=model_file_gen,
                                                   summary=summary)
        self.discriminator = self.nets.build_discriminator2D(model_shape=(self.n_rows, self.n_cols, self.c_dims),
                                                             filters=self.d_filters, k_size=self.d_ksize, drop=self.d_drop,
                                                             model_file=model_file_dis,
                                                             summary=summary)
        self.encoder = self.nets.build_encoder2D(model_shape=(self.n_rows, self.n_cols, self.c_dims), z_size=self.z_size,
                                                 filters=self.e_filters, k_size=self.e_ksize, model_file=model_file_enc,
                                                 summary=summary)
        
        # Building AE model
        input_layer = Input(shape=(self.n_rows, self.n_cols, self.c_dims), name='ae_input')
        self.model = Model(inputs=[input_layer], outputs=[self.decoder(self.encoder(input_layer))], name='ae_model')

        # define input placeholders
        self.real_img = tf.placeholder(dtype=tf.float32, name="REAL_IMG", shape=(self.batch_size, self.n_rows, self.n_cols, self.c_dims))
        self.z_noise = tf.placeholder(dtype=tf.float32, name="Z_NOISE", shape=(self.batch_size, self.z_size, ))
        self.real_labels = tf.placeholder(dtype=tf.float32, name="real_labels", shape=(self.batch_size, 1))
        self.fake_labels = tf.placeholder(dtype=tf.float32, name="fake_labels", shape=(self.batch_size, 1))
        
        # optimication of the latent space
        self.z_noise_opt = tf.get_variable("gan_z_noise_optimize", shape=(self.batch_size, self.z_size), dtype=tf.float32, initializer=tf.random_normal_initializer)
        self.z_noise_bn = (self.z_noise_opt - K.mean(self.z_noise_opt, axis=0)) / (K.var(self.z_noise_opt, axis=0) + K.epsilon())        
        
        # sinthezized images
        self.fake_img = self.decoder(self.z_noise)
        self.deco_img = self.decoder(self.encoder(self.real_img))
        self.fake_imz = self.decoder(self.z_noise_bn)       
        
        # scaling image to [0 1]
        self.real_img_scaled = (self.real_img + 1) / 2.0
        self.deco_img_scaled = (self.deco_img + 1) / 2.0
        self.fake_imz_scaled = (self.fake_imz + 1) / 2.0 
                
        self.D_real, self.D_real_logits = self.discriminator(self.real_img)
        self.D_fake, self.D_fake_logits = self.discriminator(self.fake_img)
        self.D_deco, self.D_deco_logits = self.discriminator(self.deco_img)
        self.D_fakz, self.D_fakz_logits = self.discriminator(self.fake_imz)

        """ Adversariar training lossses """
        # discriminator loss
        self.d_loss_real = self.utils.cross_entropy_loss(self.real_labels, self.D_real_logits)
        self.d_loss_fake = self.utils.cross_entropy_loss(self.fake_labels, self.D_fake_logits)
        self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2.0
        # generator loss
        self.g_loss = self.utils.cross_entropy_loss(tf.ones_like(self.D_fake), self.D_fake_logits)
        
        """ Encoder lossses """
        self.d_loss_enc = self.utils.cross_entropy_loss(tf.ones_like(self.D_deco),self.D_deco_logits)
        self.e_loss_log = 100 * self.utils.cross_entropy(self.real_img_scaled, self.deco_img_scaled)
        self.e_loss = self.e_loss_log + self.d_loss_enc
        
        """ lantent space optimization losses """         
        self.dz_loss = self.utils.cross_entropy_loss(tf.ones_like(self.D_fakz), self.D_fakz_logits)
        self.gz_loss = 100 * self.utils.cross_entropy(self.real_img_scaled, self.fake_imz_scaled)
        self.z_loss = self.gz_loss + 1.0 * self.dz_loss

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'gan_d_' in var.name]
        self.g_vars = [var for var in t_vars if 'gan_g_' in var.name]
        self.e_vars = [var for var in t_vars if 'gan_e_' in var.name]
        self.z_vars = [var for var in t_vars if 'gan_z_' in var.name]
        
        """ compute reconstruction accuracies """ 
        self.accuracy_e = self.utils.accuracy(self.real_img, self.deco_img)
        self.accuracy_z = self.utils.accuracy(self.real_img, self.fake_imz)
    
    def train(self,
              data_trn,
              data_val=None,
              epochs=1,
              best_acc=-np.inf,
              n_iterations=500,
              g_steps=2,
              patience=20,
              split=0.9,
              plots=True,
              num_plots=4,
              plot_freq=2,
              soft_labels=0.2,
              reset_model=True):
        
        self.fit_gans(data_trn, data_val, epochs, best_acc, n_iterations, g_steps, patience, split,
                      plots=plots, num_plots=num_plots, plot_freq=plot_freq, soft_labels=soft_labels, reset_model=reset_model)
        self.fit_encoder(data_trn, data_val, epochs, best_acc, patience, split,
                         plots=plots, num_plots=num_plots, plot_freq=plot_freq, reset_model=reset_model)

        
    def fit_gans(self,
                 data_trn_,
                 data_val_=None,
                 epochs=1,
                 best_acc=-np.inf,
                 n_iterations=500,
                 g_steps=2,
                 patience=20,
                 split=0.9,
                 plots=True,
                 num_plots=4,
                 plot_freq=2,
                 soft_labels=0.2,
                 reset_model=True):
        #TODO: add exeptions
        """
        optimize the discriminator and generators models using adversarial training
        """
        # Define optimizers
        d_optim = tf.compat.v1.train.AdamOptimizer(self.d_lr, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.compat.v1.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        z_optim = tf.compat.v1.train.AdamOptimizer(self.z_lr, beta1=self.beta1).minimize(self.z_loss, var_list=self.z_vars)
        
        self.sess = K.get_session()
        init_gvars = tf.variables_initializer(self.d_vars + self.g_vars)
        init_zvars = tf.variables_initializer(self.z_vars)
        if reset_model:            
            self.sess.run(init_gvars)
            print ("Initializing gans models variables ...")      
        
        # set training and validation samples
        if data_val_ is None:
            data_trn = data_trn_.copy()
            np.random.shuffle(data_trn)
            data_val = data_trn[int(split*len(data_trn)):]
            data_trn = data_trn[:int(split*len(data_trn))]
        else:
            data_trn = data_trn_.copy()
            data_val = data_val_.copy()
        
        print("Starting GANs training ...")
        labels_real = np.reshape(np.ones((self.batch_size,)), (-1, 1))
        labels_fake = np.reshape(np.zeros((self.batch_size,)), (-1, 1))
        number_of_batches = data_trn.shape[0] // self.batch_size
        patience_trn = patience
        for epoch in range(epochs):
            np.random.shuffle(data_trn)            
            loss_D, loss_G = [], []
            for index in range(0, number_of_batches):
                labels_real_noised = labels_real - np.random.uniform(0, soft_labels, (self.batch_size, 1))
                labels_fake_noised = labels_fake + np.random.uniform(0, soft_labels, (self.batch_size, 1))

                z_samples = np.random.normal(0, 1, size=[self.batch_size, self.z_size]).astype(np.float32)
                volumes_batch = data_trn[index * self.batch_size:(index + 1) * self.batch_size, :, :, :]

                # Update D network
                self.sess.run([d_optim, self.d_loss],
                              feed_dict={self.real_img: volumes_batch,
                                         self.z_noise: z_samples,
                                         self.real_labels: labels_real_noised,
                                         self.fake_labels: labels_fake_noised})

                # Update G network
                for _ in range(g_steps):                    
                    self.sess.run([g_optim, self.g_loss],
                                  feed_dict={self.real_img: volumes_batch,
                                             self.z_noise: z_samples})
            
                errD_fake, errD_real, errG = self.sess.run([self.d_loss_fake, self.d_loss_real, self.g_loss],                                                           
                                                            feed_dict={self.real_img: volumes_batch,
                                                            self.z_noise: z_samples,
                                                            self.real_labels: labels_real,
                                                            self.fake_labels: labels_fake})

                loss_D.append(errD_fake + errD_real)
                loss_G.append(errG)

                if index % (number_of_batches//4)==0 and epoch==0 and plots:
                    z_samples = np.random.normal(0, 1, size=[num_plots, self.z_size]).astype(np.float32)
                    fake_volumes = np.argmax(self.decoder.predict(z_samples), axis=-1)
                    PlotDataAE([], fake_volumes, digit_size=(self.n_rows, self.n_cols), cmap='jet', Only_Result=False, num=num_plots)

            print("Epoch ->", epoch+1, "D loss -> ", np.mean(loss_D), "G loss -> ", np.mean(loss_G))
            if epoch % plot_freq==0 and plots:
                z_samples = np.random.normal(0, 1, size=[num_plots, self.z_size]).astype(np.float32)
                fake_volumes = np.argmax(self.decoder.predict(z_samples), axis=-1)
                PlotDataAE([], fake_volumes, digit_size=(self.n_rows, self.n_cols), cmap='jet', Only_Result=False, num=num_plots)

            self.sess.run(init_zvars)
            np.random.shuffle(data_val)
            for _ in range(n_iterations):
                _, d_loss_, g_loss_, acc_ = self.sess.run([z_optim, self.dz_loss, self.gz_loss, self.accuracy_z],
                                                          feed_dict={self.real_img: data_val[:self.batch_size]})
            print('\n')
            print('D loss ->', d_loss_, 'G loss ->', g_loss_, 'reconstruction acc ->', acc_.mean())

            if 100 * acc_.mean() - d_loss_ > best_acc:
                best_acc = 100 * acc_.mean() - d_loss_
                patience_trn = patience
                Save_Model(self.decoder, self.saving_path + self.decoder_name)
                Save_Model(self.discriminator, self.saving_path + self.discriminator_name)
                print('Saving check point ...')
            else:
                patience_trn -= 1
            if patience_trn < 0:
                print('Training stoped ...')
                break
            if epoch < 5:
                best_acc = -np.inf
        print('Restauring best Discriminator and Generator Models ...')
        self.build_model(model_file=self.saving_path+self.name_base, model_file_enc=None, summary=False)

    def fit_encoder(self,
                    data_trn_,
                    data_val_=None,
                    epochs=1,
                    best_acc=-np.inf,
                    patience=20,
                    split=0.9,
                    plots=True,
                    num_plots=4,
                    plot_freq=2,
                    reset_model=True
                    ):
        #TODO: Delete models after training
        # Learning parameters must be defined here
        e_optim = tf.compat.v1.train.AdamOptimizer(self.e_lr, beta1=self.beta1).minimize(self.e_loss, var_list=self.e_vars)
        self.sess = K.get_session()
        init_op = tf.variables_initializer(self.e_vars)
        if reset_model:
            self.build_model(model_file=self.saving_path+self.name_base, model_file_enc=None, summary=False)
            e_optim = tf.compat.v1.train.AdamOptimizer(self.e_lr, beta1=self.beta1).minimize(self.e_loss, var_list=self.e_vars)
            self.sess = K.get_session()
            init_op = tf.variables_initializer(self.e_vars)
            self.sess.run(init_op)
            print ("Initializing encoder variables ...")

        # Setting training and validation samples
        if data_val_ is None:
            data_trn = data_trn_.copy()
            np.random.shuffle(data_trn)
            data_val = data_trn[int(split*len(data_trn)):]
            data_trn = data_trn[:int(split*len(data_trn))]
        else:
            data_trn = data_trn_.copy()
            data_val = data_val_.copy()

        print("Starting Encoder training ...")
        patience_trn = patience
        for epoch in range(10*epochs):
            np.random.shuffle(data_trn)
            number_of_batches = data_trn.shape[0] // self.batch_size
            for index in range(0, number_of_batches):
                volumes_batch = data_trn[index * self.batch_size:(index + 1) * self.batch_size]
                self.sess.run([e_optim], 
                              feed_dict={self.real_img: volumes_batch})
            # Validation
            number_of_batches = data_val.shape[0] // self.batch_size
            loss_E, loss_D, Acc = [], [], []
            for index in range(0, number_of_batches):
                volumes_batch = data_val[index * self.batch_size:(index + 1) * self.batch_size]
                errE, errD, acc_e, deco_imgs = self.sess.run([self.e_loss, self.d_loss_enc, self.accuracy_e, self.deco_img],
                                                             feed_dict={self.real_img: volumes_batch})
            
                loss_E.append(errE)
                loss_D.append(errD)
                Acc.append(acc_e)

            if np.mean(Acc) > best_acc:
                best_acc = np.mean(Acc)
                patience_trn = patience
                if self.saving_path:
                    print("Saving best model ...")
                    Save_Model(self.encoder, self.saving_path + self.encoder_name)
            else:
                patience_trn -= 1
            if patience_trn < 0:
                print('Training stoped ...')
                break

            if plots and epoch % plot_freq:
                real_samples = data_trn[:num_plots]
                deco_samples = self.decoder.predict(self.encoder.predict(real_samples))
                real_samples = np.argmax(real_samples, axis=-1)
                deco_samples = np.argmax(deco_samples, axis=-1)
                diff = real_samples - deco_samples
                PlotDataAE([], diff, digit_size=(self.n_rows, self.n_cols), Only_Result=False, num=num_plots)
            print('epoch', epoch+1, 'D loss ->', np.mean(loss_D), 'E loss ->', np.mean(loss_E), 'Acc ->', np.mean(Acc))
    
    def Encoder(self, x_test):
        """
        Return predicted result from the encoder model
        """
        return self.encoder.predict(x_test)
    
    def Decoder(self, x_test, binary=False):
        """
        Return predicted result from the AE model
        """
        if binary:
            return np.argmax(self.model.predict(x_test), axis=-1)
        return self.model.predict(x_test)
        
    def generate(self, number_latent_sample=20,std=1,binary=False):
        """
        Generating examples from samples from the latent distribution.
        """
        latent_sample = np.random.normal(0, std, size=(number_latent_sample, self.z_size))
        if binary:
            return np.argmax(self.decoder.predict(latent_sample),axis=-1)
        return self.decoder.predict(latent_sample)
    
    def load_model(self, model_file, summary=True):
        model = self.utils.build_pretrained_model(model_file)
        if summary:
            model.summary()
        return model


class WGAN2D_MPS(object):
    def __init__(self, input_shape,
                 d_filters=32, g_filters=32, e_filters=32,
                 d_ksize=5, g_ksize=4, e_ksize=4, z_size=500, batch_size=64,
                 d_lr=5e-5, g_lr=5e-5, e_lr=0.001, z_lr=0.1, beta1=0.5, clip_value=0.05,
                 model_file=None, saving_path=None, name='geo_WGAN_', summary=True
                ):

        if saving_path is None:
            saving_path = './'
        self.saving_path = saving_path
        self.n_rows = input_shape[0]
        self.n_cols = input_shape[1]
        self.c_dims = input_shape[2]
        self.name_base = name+'z_'+str(z_size)+'_'+str(self.n_rows)+'x'+str(self.n_cols)
        self.discriminator_name = self.name_base + '_discriminator'
        self.decoder_name = self.name_base + '_decoder'
        self.encoder_name = self.name_base + '_encoder'
        self.batch_size = batch_size        
        self.d_filters=d_filters
        self.g_filters=g_filters
        self.e_filters=e_filters
        self.d_ksize=d_ksize
        self.g_ksize=g_ksize
        self.e_ksize=e_ksize
        self.z_size = z_size        
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.e_lr = e_lr
        self.z_lr = z_lr
        self.beta1 = beta1
        self.clip_value = clip_value
        self.utils = Utilities()
        self.nets = Networks()
        self.model_file = model_file
        self.summary = summary
        self.build_model(self.model_file, summary=self.summary)
    
    def build_model(self, model_file=None, model_file_enc=None, summary=False):
        
        if model_file is not None:
            model_file_dis=model_file+'_discriminator'
            model_file_gen=model_file+'_decoder'
        else:
            model_file_dis=model_file
            model_file_gen=model_file

        # build network models
        K.clear_session()
        self.decoder = self.nets.build_generator2D(model_shape=(self.n_rows, self.n_cols, self.c_dims), z_size=self.z_size,
                                                   filters=self.g_filters, k_size=self.g_ksize, model_file=model_file_gen,
                                                   summary=summary)
        self.discriminator = self.nets.build_discriminator2D(model_shape=(self.n_rows, self.n_cols, self.c_dims),
                                                             filters=self.d_filters, k_size=self.d_ksize, model_file=model_file_dis,
                                                             drop=False, summary=summary)
        self.encoder = self.nets.build_encoder2D(model_shape=(self.n_rows, self.n_cols, self.c_dims), z_size=self.z_size,
                                                 filters=self.e_filters, k_size=self.e_ksize, model_file=model_file_enc,
                                                 summary=summary)
        
        # Building AE model
        input_layer = Input(shape=(self.n_rows, self.n_cols, self.c_dims), name='ae_input')
        self.model = Model(inputs=[input_layer], outputs=[self.decoder(self.encoder(input_layer))], name='ae_model')

        # define input placeholders
        self.real_img = tf.placeholder(dtype=tf.float32, name="REAL_IMG", shape=(self.batch_size, self.n_rows, self.n_cols, self.c_dims))
        self.z_noise = tf.placeholder(dtype=tf.float32, name="Z_NOISE", shape=(self.batch_size, self.z_size, ))
        
        # optimication of the latent space
        self.z_noise_opt = tf.get_variable("gan_z_noise_optimize", shape=(self.batch_size, self.z_size), dtype=tf.float32, initializer=tf.random_normal_initializer)
        self.z_noise_bn = (self.z_noise_opt - K.mean(self.z_noise_opt, axis=0)) / (K.var(self.z_noise_opt, axis=0) + K.epsilon())        
        
        # sinthezized images
        self.fake_img = self.decoder(self.z_noise)
        self.deco_img = self.decoder(self.encoder(self.real_img))
        self.fake_imz = self.decoder(self.z_noise_bn)       
        
        # scaling image to [0 1]
        self.real_img_scaled = (self.real_img + 1) / 2.0
        self.deco_img_scaled = (self.deco_img + 1) / 2.0
        self.fake_imz_scaled = (self.fake_imz + 1) / 2.0 
                
        self.D_real, self.D_real_logits = self.discriminator(self.real_img)
        self.D_fake, self.D_fake_logits = self.discriminator(self.fake_img)
        self.D_deco, self.D_deco_logits = self.discriminator(self.deco_img)
        self.D_fakz, self.D_fakz_logits = self.discriminator(self.fake_imz)

        """ Adversariar training lossses """
        # Discriminator loss
        self.d_loss = tf.reduce_mean(self.D_fake_logits) - tf.reduce_mean(self.D_real_logits)
        # Generator loss
        self.g_loss = -tf.reduce_mean(self.D_fake_logits)
        
        """ Encoder lossses """
        # self.d_loss_enc = self.utils.cross_entropy_loss(tf.ones_like(self.D_deco),self.D_deco_logits)
        self.e_loss_log = 100 * tf.reduce_mean(self.utils.cross_entropy(self.real_img_scaled, self.deco_img_scaled))
        # self.e_loss = self.e_loss_log + self.d_loss_enc
        self.e_loss = self.e_loss_log
        
        """ lantent space optimization losses """
        # self.dz_loss = self.utils.cross_entropy_loss(tf.ones_like(self.D_fakz), self.D_fakz_logits)
        self.gz_loss = 1000 * tf.reduce_mean(self.utils.cross_entropy(self.real_img_scaled, self.fake_imz_scaled))
        # self.z_loss = self.gz_loss + 1.0 * self.dz_loss
        self.z_loss = self.gz_loss

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'gan_d_' in var.name]
        self.g_vars = [var for var in t_vars if 'gan_g_' in var.name]
        self.e_vars = [var for var in t_vars if 'gan_e_' in var.name]
        self.z_vars = [var for var in t_vars if 'gan_z_' in var.name]
        
        """ compute reconstruction accuracies """ 
        self.accuracy_e = self.utils.accuracy(self.real_img, self.deco_img)
        self.accuracy_z = self.utils.accuracy(self.real_img, self.fake_imz)
    
    def train(self,
              data_trn,
              data_val=None,
              epochs=1,
              best_acc=-np.inf,
              n_iterations=500,
              patience=20,
              split=0.9,
              plots=True,
              num_plots=4,
              plot_freq=2,
              reset_model=True):
        
        self.fit_gans(data_trn, data_val, epochs, best_acc, n_iterations, patience, split,
                      plots=plots, num_plots=num_plots, plot_freq=plot_freq, reset_model=reset_model)
        self.fit_encoder(data_trn, data_val, epochs, best_acc, patience, split,
                         plots=plots, num_plots=num_plots, plot_freq=plot_freq, reset_model=reset_model)
        
    def fit_gans(self,
                 data_trn_,
                 data_val_=None,
                 epochs=1,
                 best_acc=-np.inf,
                 n_iterations=500,
                 patience=20,
                 split=0.9,
                 plots=True,
                 num_plots=4,
                 plot_freq=2,
                 th_acc = 90,
                 reset_model=True):
        """
        optimize the discriminator and generators models using adversarial training
        """
        # Define optimizers
        d_optim = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.d_lr).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.g_lr).minimize(self.g_loss, var_list=self.g_vars)
        z_optim = tf.compat.v1.train.AdamOptimizer(self.z_lr, beta1=self.beta1).minimize(self.z_loss, var_list=self.z_vars)

        clip_updates = [w.assign(tf.clip_by_value(w, -self.clip_value, self.clip_value)) for w in self.d_vars]
        
        self.sess = K.get_session()
        init_gvars = tf.variables_initializer(self.d_vars + self.g_vars)
        init_zvars = tf.variables_initializer(self.z_vars)
        if reset_model:            
            self.sess.run(init_gvars)
            print ("Initializing GANs models variables ...")      
        
        # Setting training and validation samples
        if data_val_ is None:
            data_trn = data_trn_.copy()
            np.random.shuffle(data_trn)
            data_val = data_trn[int(split*len(data_trn)):]
            data_trn = data_trn[:int(split*len(data_trn))]
        else:
            data_trn = data_trn_.copy()
            data_val = data_val_.copy()            
        
        print("Starting GANs training ...")
        g_step = 0
        early_stop = False
        loss_D, loss_G = [], []
        best_loss_D = np.inf
        patience_trn = patience
        iterations = 5000
        saved_checkpoint = False
        for iteration in range(iterations*epochs):
            #  Train Discriminator 
            n_critic = 100 if g_step < 25 or (g_step + 1) % 500 == 0 else 5
            for _ in range(n_critic):
                z_samples = np.random.normal(0, 1, size=[self.batch_size, self.z_size]).astype(np.float32)
                # Select a random batch of images
                idx = np.random.choice(data_trn.shape[0], self.batch_size, replace=False)
                imgs = data_trn[idx]
                # Update D network
                _, d_loss = self.sess.run([d_optim, self.d_loss],
                                          feed_dict={self.real_img: imgs,
                                                     self.z_noise: z_samples})
                self.sess.run(clip_updates)
                loss_D.append(d_loss)    
            
            #  Train Generator
            z_samples = np.random.normal(0, 1, size=[self.batch_size, self.z_size]).astype(np.float32)
            # Update G network
            _, g_loss = self.sess.run([g_optim, self.g_loss],
                                      feed_dict={self.z_noise: z_samples})
            g_step += 1
            loss_G.append(g_loss)
            
            if (iteration+1) % (iterations*plot_freq) == 0:
                if plots:
                    print('Random realizations ...')
                    z_samples = np.random.normal(0, 1, size=[num_plots, self.z_size]).astype(np.float32)
                    x_generated = np.argmax(self.decoder.predict(z_samples), axis=-1)
                    PlotDataAE([], x_generated, digit_size=(self.n_rows, self.n_cols), cmap='jet', Only_Result=False, num=num_plots)
                    
            # Reconstruction accuracy
            if (iteration+1) % (iterations) == 0:
                # Select a random batch of images
                idx = np.random.randint(0, data_val.shape[0], self.batch_size)
                imgs = data_val[idx]
                self.sess.run(init_zvars)
                np.random.shuffle(data_val)
                for _ in range(n_iterations):                
                    _, g_loss_, acc_ = self.sess.run([z_optim, self.gz_loss, self.accuracy_z],
                                                        feed_dict={self.real_img: data_val[:self.batch_size]})
                
                # print('Iteration: ', iteration+1, '  G loss: ', g_loss_, '  Accuracy: ', acc_.mean())
                if 100*np.mean(acc_) >= th_acc:
                    early_stop = True
                if early_stop:    
                    if (best_loss_D > np.abs(np.mean(loss_D))) and 100*np.mean(acc_) >= th_acc:
                        best_loss_D = np.abs(np.mean(loss_D))
                        patience_trn = patience
                        Save_Model(self.discriminator, self.saving_path + self.discriminator_name)
                        Save_Model(self.decoder, self.saving_path + self.decoder_name)
                        print('Saving check point ...')
                        saved_checkpoint = True
                    else:
                        patience_trn -= 1
                    if patience_trn < 0:
                        print('Training stoped ...')
                        break
                
                print('Iteration: ', iteration+1, '  D loss: ', np.mean(loss_D), '  G loss: ', np.mean(loss_G), '  Accuracy: ', acc_.mean())
                loss_D, loss_G = [], []
        if saved_checkpoint is False:
            Save_Model(self.discriminator, self.saving_path + self.discriminator_name)
            Save_Model(self.decoder, self.saving_path + self.decoder_name)
            print('Saving check point, last iteration')
        print('Restauring best Discriminator and Generator Models ...')
        self.build_model(model_file=self.saving_path+self.name_base, model_file_enc=None, summary=False)

    def fit_encoder(self,
                    data_trn,
                    data_val=None,
                    epochs=1,
                    best_acc=-np.inf,
                    patience=50,
                    split=0.9,
                    plots=True,
                    num_plots=4,
                    plot_freq=2,
                    reset_model=True
                    ):

        e_optim = tf.compat.v1.train.AdamOptimizer(self.e_lr, beta1=self.beta1).minimize(self.e_loss, var_list=self.e_vars)
        self.sess = K.get_session()
        init_op = tf.variables_initializer(self.e_vars)
        if reset_model:
            self.build_model(model_file=self.saving_path+self.name_base, model_file_enc=None, summary=False)
            e_optim = tf.compat.v1.train.AdamOptimizer(self.e_lr, beta1=self.beta1).minimize(self.e_loss, var_list=self.e_vars)
            self.sess = K.get_session()
            init_op = tf.variables_initializer(self.e_vars)
            self.sess.run(init_op)
            print ("Initializing encoder variables ...")

        # Setting training and validation samples
        if data_val is None:
            np.random.shuffle(data_trn)
            data_val = data_trn[int(split*len(data_trn)):]
            data_trn = data_trn[:int(split*len(data_trn))]

        print("Starting Encoder training ...")
        patience_trn = patience
        for epoch in range(10*epochs):
            np.random.shuffle(data_trn)
            number_of_batches = data_trn.shape[0] // self.batch_size
            for index in range(0, number_of_batches):
                volumes_batch = data_trn[index * self.batch_size:(index + 1) * self.batch_size]
                self.sess.run([e_optim], 
                              feed_dict={self.real_img: volumes_batch})
            # Validation
            number_of_batches = data_val.shape[0] // self.batch_size
            loss_E, loss_D, Acc = [], [], []
            for index in range(0, number_of_batches):
                volumes_batch = data_val[index * self.batch_size:(index + 1) * self.batch_size]
                errE, acc_e, deco_imgs = self.sess.run([self.e_loss, self.accuracy_e, self.deco_img],
                                                             feed_dict={self.real_img: volumes_batch})
                loss_E.append(errE)
                # loss_D.append(errD)
                Acc.append(acc_e)

            if np.mean(Acc) > best_acc:
                best_acc = np.mean(Acc)
                patience_trn = patience
                if self.saving_path:
                    Save_Model(self.encoder, self.saving_path + self.encoder_name)
                if plots and index % plot_freq:
                    real_samples = data_trn[:num_plots]
                    deco_samples = self.decoder.predict(self.encoder.predict(real_samples))
                    real_samples = np.argmax(real_samples, axis=-1)
                    deco_samples = np.argmax(deco_samples, axis=-1)
                    diff = np.abs(volumes_batch - deco_imgs)
                    PlotDataAE([], diff, digit_size=(self.n_rows, self.n_cols), Only_Result=False, num=num_plots)                
            else:
                patience_trn -= 1
            if patience_trn < 0:
                print('Training stoped using early stop ...')
                break
            print('epoch', epoch+1, 'E loss ->', np.mean(loss_E), 'Acc ->', np.mean(Acc))
            # print('epoch', epoch+1, 'D loss ->', np.mean(loss_D), 'E loss ->', np.mean(loss_E), 'Acc ->', np.mean(Acc))
        print("Training stoped, last epoch reached ... ")

    def Encoder(self, x_test):
        """
        Return predicted result from the encoder model
        """
        return self.encoder.predict(x_test)
    
    def Decoder(self, x_test, binary=False):
        """
        Return predicted result from the AE model
        """
        if binary:
            return np.argmax(self.model.predict(x_test), axis=-1)
        return self.model.predict(x_test)
        
    def generate(self, number_latent_sample=20, std=1, binary=False):
        """
        Generating examples from samples from the latent distribution.
        """
        latent_sample = np.random.normal(0, std, size=(number_latent_sample, self.z_size))
        if binary:
            return np.argmax(self.decoder.predict(latent_sample),axis=-1)
        return self.decoder.predict(latent_sample)

    def load_model(self, model_file, summary=True):
        model = self.utils.build_pretrained_model(model_file)
        if summary:
            model.summary()
        return model


class AlphaGAN_MPS(object):
    def __init__(self, input_shape,
                d_filters=32, g_filters=32, e_filters=32, c_filters=3500,
                d_ksize=5, g_ksize=5, e_ksize=5, z_size=500, batch_size=32,
                d_lr=0.0002, g_lr=0.0002, e_lr=0.0002, c_lr=0.0002, beta1=0.5, alpha=1.0,
                model_file=None, saving_path=None, name='geo_AlphaGAN_', summary=True
                ):

        self.saving_path = saving_path
        self.n_rows = input_shape[0]
        self.n_cols = input_shape[1]
        self.c_dims = input_shape[2]
        self.discriminator_name = name+'z_'+str(z_size)+'_'+str(self.n_rows)+'x'+str(self.n_cols)+'_discriminator'
        self.decoder_name = name+'z_'+str(z_size)+'_'+str(self.n_rows)+'x'+str(self.n_cols)+'_decoder'
        self.encoder_name = name+'z_'+str(z_size)+'_'+str(self.n_rows)+'x'+str(self.n_cols)+'_encoder'
        self.code_discriminator_name = name+'z_'+str(z_size)+'_'+str(self.n_rows)+'x'+str(self.n_cols)+'_code_discriminator'
        self.batch_size = batch_size        
        self.d_filters=d_filters
        self.g_filters=g_filters
        self.e_filters=e_filters
        self.c_filters=c_filters
        self.d_ksize=d_ksize
        self.g_ksize=g_ksize
        self.e_ksize=e_ksize
        self.z_size = z_size        
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.e_lr = e_lr
        self.c_lr = c_lr
        self.beta1 = beta1
        self.alpha = alpha
        self.utils = Utilities()
        self.nets = Networks()   
        self.build_model(model_file, summary=summary)        
    
    def build_model(self,
                    model_file=None,
                    summary=False
                    ):
        if model_file is not None:
            model_file_dis=model_file+'_discriminator'
            model_file_dis_code=model_file+'_code_discriminator'
            model_file_gen=model_file+'_decoder'
            model_file_enc=model_file+'_encoder'
        else:
            model_file_dis=model_file
            model_file_dis_code=model_file
            model_file_gen=model_file
            model_file_enc=model_file

        # build network models
        K.clear_session()
        self.decoder = self.nets.build_generator2D(model_shape=(self.n_rows, self.n_cols, self.c_dims), z_size=self.z_size,
                                                   filters=self.g_filters, k_size=self.g_ksize, model_file=model_file_gen,
                                                   summary=summary)
        self.discriminator = self.nets.build_discriminator2D(model_shape=(self.n_rows, self.n_cols, self.c_dims),
                                                             filters=self.d_filters, k_size=self.d_ksize, drop=False, model_file=model_file_dis,
                                                             summary=summary)
        self.encoder = self.nets.build_encoder2D(model_shape=(self.n_rows, self.n_cols, self.c_dims), z_size=self.z_size,
                                                 filters=self.e_filters, k_size=self.e_ksize, bn=False, model_file=model_file_enc,
                                                 summary=summary)
        self.code_discriminator = self.nets.build_code_discriminator(model_file=model_file_dis_code,
                                                                     filters=self.c_filters,z_size=self.z_size,
                                                                     summary=summary)
        
        # Building AE model
        input_layer = Input(shape=(self.n_rows, self.n_cols, self.c_dims), name='ae_input')
        self.model = Model(inputs=[input_layer], outputs=[self.decoder(self.encoder(input_layer))], name='ae_model')

        # define input placeholders
        self.real_img = tf.placeholder(dtype=tf.float32, name="REAL_IMG", shape=(self.batch_size, self.n_rows, self.n_cols, self.c_dims))
        self.z_noise = tf.placeholder(dtype=tf.float32, name="Z_NOISE", shape=(self.batch_size, self.z_size, ))
        
        # Encode to lantent space
        self.z_latent = self.encoder(self.real_img)
        
        self.C_real, self.C_real_logits = self.code_discriminator(self.z_noise)
        self.C_fake, self.C_fake_logits = self.code_discriminator(self.z_latent)
        
        # Synthezized images
        self.fake_img = self.decoder(self.z_noise)
        self.deco_img = self.decoder(self.z_latent)
                
        self.D_real, self.D_real_logits = self.discriminator(self.real_img)
        self.D_fake, self.D_fake_logits = self.discriminator(self.fake_img)
        self.D_deco, self.D_deco_logits = self.discriminator(self.deco_img)

        """ Reconstruction loss """
        self.l1_loss = self.alpha * self.utils.reconstruction_loss(self.real_img, self.deco_img)

        """ Discriminator losses """
        self.d_loss_real = self.utils.cross_entropy_loss(tf.ones_like (self.D_real), self.D_real_logits)
        self.d_loss_fake = self.utils.cross_entropy_loss(tf.zeros_like(self.D_fake), self.D_fake_logits)
        self.d_loss_deco = self.utils.cross_entropy_loss(tf.zeros_like(self.D_deco), self.D_deco_logits)
        self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_deco
        
        self.c_loss_real = self.utils.cross_entropy_loss(tf.ones_like (self.C_real), self.C_real_logits)
        self.c_loss_fake = self.utils.cross_entropy_loss(tf.zeros_like(self.C_fake), self.C_fake_logits)
        self.c_loss = self.c_loss_real + self.c_loss_fake
        
        """ Generator loss """
        # TODO: Check this
        self.g_loss = self.l1_loss + self.RD_phi(self.D_deco_logits) + self.RD_phi(self.D_fake_logits)
        # self.g_loss = self.l1_loss + self.RD_phi2(self.D_deco) + self.RD_phi2(self.D_fake)
                
        """ Encoder losss """
        # TODO: Check this
        self.e_loss = self.l1_loss + self.RC_w(self.C_fake_logits)
        # self.e_loss = self.l1_loss + self.RC_w2(self.C_fake)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'gan_d_' in var.name]
        self.g_vars = [var for var in t_vars if 'gan_g_' in var.name]
        self.e_vars = [var for var in t_vars if 'gan_e_' in var.name]
        self.c_vars = [var for var in t_vars if 'gan_c_' in var.name]
        
        """ compute reconstruction accuracy """ 
        self.accuracy = self.utils.accuracy(self.real_img, self.deco_img)
    
    def train(self,
              data_trn_,
              data_val_=None,
              epochs=1,
              best_acc=-np.inf,
              patience=50,
              split=0.9,
              plots=True,
              num_plots=4,
              plot_freq=2,
              reset_model=True):
        """
        optimize the discriminator and generators models using adversarial training
        """
        # Define optimizers
        d_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        e_optim = tf.train.AdamOptimizer(self.e_lr, beta1=self.beta1).minimize(self.e_loss, var_list=self.e_vars)
        c_optim = tf.train.AdamOptimizer(self.c_lr, beta1=self.beta1).minimize(self.c_loss, var_list=self.c_vars)
        
        self.sess = K.get_session()
        init_op = tf.global_variables_initializer()
        if reset_model:            
            self.sess.run(init_op)
            print ("Initializing GANs models variables ...")      
        
        # Setting training and validation samples
        if data_val_ is None:
            data_trn = data_trn_.copy()
            np.random.shuffle(data_trn)
            data_val = data_trn[int(split*len(data_trn)):]
            data_trn = data_trn[:int(split*len(data_trn))]
        else:
            data_trn = data_trn_.copy()
            data_val = data_val_.copy()
            
        patience_trn = patience
        print("Starting GANs training ...")
        for epoch in range(epochs):
            np.random.shuffle(data_trn)
            np.random.shuffle(data_val)
            batch_idxs = data_trn.shape[0] // self.batch_size
            loss_E, loss_G, loss_D, loss_C = [], [], [], []

            for idx in range(0, batch_idxs):
                # Sample noise from a normal distribuition
                z_samples = np.random.normal(0, 1, size=[self.batch_size, self.z_size]).astype(np.float32)
                volumes_batch = data_trn[idx * self.batch_size:(idx + 1) * self.batch_size, :, :, :]

                # 1. Update Encoder:
                _, loss_e = self.sess.run([e_optim, self.e_loss],
                                          feed_dict={self.real_img: volumes_batch,
                                                     self.z_noise: z_samples})
                # 2. Update Generator:
                _, loss_g = self.sess.run([g_optim, self.g_loss],
                                     feed_dict={self.real_img: volumes_batch,
                                                self.z_noise: z_samples})
                # 3. Update Discriminator phi
                _, loss_d = self.sess.run([d_optim, self.d_loss],
                                     feed_dict={self.real_img: volumes_batch,
                                                self.z_noise: z_samples})
                # 4. Update Code Discriminator
                _, loss_c = self.sess.run([c_optim, self.c_loss],
                                     feed_dict={self.real_img: volumes_batch,
                                                self.z_noise: z_samples})

                loss_E.append(loss_e) 
                loss_G.append(loss_g)
                loss_D.append(loss_d)
                loss_C.append(loss_c)
                
                if idx % (batch_idxs//4)==0 and epoch < 2 and plots:
                    samples = data_trn[:num_plots]
                    zsamples = np.random.normal(0, 1, size=[num_plots, self.z_size]).astype(np.float32)
                    fake_volumes = self.decoder.predict(zsamples)
                    PlotDataAE([],np.argmax(fake_volumes, axis=-1), digit_size=(self.n_rows, self.n_cols), cmap='jet', Only_Result=False, num=num_plots)
                    # print("Epoch ->", epoch+1, "E loss -> ", np.mean(loss_E), "G loss -> ", np.mean(loss_G), "D loss -> ", np.mean(loss_D), "C loss -> ", np.mean(loss_C))

            print("Epoch ->", epoch+1, "E loss -> ", np.mean(loss_E), "G loss -> ", np.mean(loss_G), "D loss -> ", np.mean(loss_D), "C loss -> ", np.mean(loss_C))
            
            if epoch % plot_freq==0 and plots:
                print('Random realizations ...')
                zsamples = np.random.normal(0, 1, size=[num_plots, self.z_size]).astype(np.float32)
                fake_volumes = self.decoder.predict(zsamples)
                PlotDataAE([],np.argmax(fake_volumes, axis=-1), digit_size=(self.n_rows, self.n_cols), cmap='jet', Only_Result=False, num=num_plots)
                    
            with self.sess.as_default():
                testing_acc = []
                for i in range(len(data_val)//self.batch_size):
                    x_test_batch = data_val[i * self.batch_size:(i + 1) * self.batch_size, :, :, :]
                    testing_acc.append(self.accuracy.eval({self.real_img: x_test_batch}))
                test_acc = np.mean(testing_acc)
                print('Acc -->', test_acc)

                if best_acc < test_acc and epoch > 9:
                    best_acc = test_acc.copy()
                    patience_trn = patience
                    if self.saving_path:
                        print('Saving best models ...')
                        Save_Model(self.encoder, self.saving_path+self.encoder_name)
                        Save_Model(self.decoder, self.saving_path+self.decoder_name)
                        Save_Model(self.discriminator, self.saving_path+self.discriminator_name)
                        Save_Model(self.code_discriminator, self.saving_path+self.code_discriminator_name)
                    real_samples = data_val[:num_plots]
                    x_hat = self.decoder.predict(self.encoder.predict(real_samples))
                    PlotDataAE(np.argmax(real_samples, axis=-1), np.argmax(x_hat, axis=-1), digit_size=(self.n_rows, self.n_cols), cmap='jet', Only_Result=True, num=num_plots)    
                else:
                    patience_trn -= 1
                if patience_trn < 0:
                    print('Training stoped ...')
                    break
    
    def RD_phi(self, D_x_logits):
        loss = self.utils.cross_entropy_loss(tf.ones_like(D_x_logits), D_x_logits)
        return loss
    
    def RD_phi2(self, D_x):
        loss = -K.log(K.clip(D_x, 1e-07, 1.0)) + K.log(K.clip(1 - D_x, 1e-07, 1.0))
        return loss

    def RC_w(self, C_z_logits):
        loss = self.utils.cross_entropy_loss(tf.ones_like(C_z_logits), C_z_logits)
        return loss
    
    def RC_w2(self, C_z):
        loss = -K.log(K.clip(C_z, 1e-07, 1.0)) + K.log(K.clip(1 - C_z, 1e-07, 1.0))
        return loss
    
    def Encoder(self, x_test):
        """
        Return predicted result from the encoder model
        """
        return self.encoder.predict(x_test)
    
    def Decoder(self, x_test, binary=False):
        """
        Return predicted result from the AE model
        """
        if binary:
            return np.argmax(self.model.predict(x_test), axis=-1)
        return self.model.predict(x_test)
        
    def generate(self, number_latent_sample=20, std=1, binary=False):
        """
        Generating examples from samples from the latent distribution.
        """
        latent_sample = np.random.normal(0, std, size=(number_latent_sample, self.z_size))
        if binary:
            return np.argmax(self.decoder.predict(latent_sample),axis=-1)
        return self.decoder.predict(latent_sample)
    
    def load_model(self, model_file, summary=True):
        model = self.utils.build_pretrained_model(model_file)
        if summary:
            model.summary()
        return model


class CycleGAN_MPS(object):
    def __init__(self, input_shape,
                filters=32, batch_size=4, epsilon=0.3, Nr=5000, Nt=5000,
                d_lr_pca=0.0002, d_lr_bin=0.0002, g_lr_pca=0.0002, g_lr_bin=0.0002,
                d_ksize_pca=4, d_ksize_bin=4, g_ksize_pca=4, g_ksize_bin=4, beta1=0.5, alpha=10, d_drop=False,
                model_file=None, saving_path=None, name='geo_CycleGAN_', summary=True
                ):
        # TO-DO: mirar lo del z 
        self.saving_path = saving_path
        self.n_rows = input_shape[0]
        self.n_cols = input_shape[1]
        self.c_dims = input_shape[2]
        self.name_base = name+'e_'+str(epsilon)+'_'+str(self.n_rows)+'x'+str(self.n_cols)
        self.discriminator_name_pca = self.name_base + '_pca_discriminator'
        self.discriminator_name_bin = self.name_base + '_bin_discriminator'
        self.decoder_name_pca = self.name_base + '_pca_decoder'
        self.decoder_name_bin = self.name_base + '_bin_decoder'
        self.batch_size = batch_size
        self.filters = filters
        self.alpha = alpha
        self.d_lr_pca = d_lr_pca
        self.d_lr_bin = d_lr_bin
        self.g_lr_pca = g_lr_pca
        self.g_lr_bin = g_lr_bin
        self.d_ksize_pca = d_ksize_pca
        self.d_ksize_bin = d_ksize_bin
        self.g_ksize_pca = g_ksize_pca
        self.g_ksize_bin = g_ksize_bin
        self.d_drop = d_drop
        self.beta1 = beta1
        self.epsilon = epsilon
        self.Nr = Nr
        self.Nt = Nt
        self.utils = Utilities()
        self.nets = Networks()
        self.build_model(model_file=model_file,
                         summary=summary)
    
    # TODO: pass to utilities
    def build_pca_model(self, data, plots=True):
        # To-Do Build method to accept different Nr and Nt
        print('Building pca model ...')
        data = np.expand_dims(np.argmax(data, axis=-1), axis=-1)
        trn_data, val_data = self.utils.split_data(data, self.Nr)
        pca_model = ComputePCA(trn_data, epsilon=self.epsilon, computeAll=True)
        pca_realizations = self.utils.random_pca_realizations((self.n_rows, self.n_cols), pca_model, self.Nt, plots=plots)
        trn_data_pca = self.utils.map2pca(trn_data, (self.n_rows, self.n_cols), pca_model, plots=plots)
        val_data_pca = self.utils.map2pca(val_data, (self.n_rows, self.n_cols), pca_model, plots=plots)
        trn_data_bin = to_categorical(trn_data)
        val_data_bin = to_categorical(val_data)
        trn_data_bin = 2 * trn_data_bin - 1
        val_data_bin = 2 * val_data_bin - 1
        
        # # scaling to -1 -- 1
        # pca_realizations = self.min_max_samples(pca_realizations)
        # trn_data_pca = self.min_max_samples(trn_data_pca)
        # val_data_pca = self.min_max_samples(val_data_pca)
        
        # trn_model_pca = np.concatenate((trn_data_pca[:self.Nt//5], pca_realizations[:4*self.Nt//5]), axis=0)
        Data_trn = np.concatenate((pca_realizations, trn_data_bin), axis=-1)
        Data_val = np.concatenate((val_data_pca, val_data_bin), axis=-1)
        Data_trn_paired = Data_val[:self.Nr]
        Data_val_paired = Data_val[self.Nr:2*self.Nr]
        return Data_trn, Data_trn_paired, Data_val_paired
    
    # def build_pca_model(self, data, plots=True):
    #     # To-Do Build method to accept different Nr and Nt
    #     print('Building pca model ...')
    #     data = np.expand_dims(np.argmax(data, axis=-1), axis=-1)
    #     trn_data, val_data = self.utils.split_data(data, self.Nr)
    #     pca_model = ComputePCA(trn_data, epsilon=self.epsilon, computeAll=True)
    #     pca_realizations = self.utils.random_pca_realizations((self.n_rows, self.n_cols), pca_model, self.Nt, plots=plots)
    #     trn_data_pca = self.utils.map2pca(trn_data, (self.n_rows, self.n_cols), pca_model, plots=plots)
    #     val_data_pca = self.utils.map2pca(val_data, (self.n_rows, self.n_cols), pca_model, plots=plots)
    #     trn_data_bin = to_categorical(trn_data)
    #     val_data_bin = to_categorical(val_data)
    #     trn_data_bin = 2 * trn_data_bin - 1
    #     val_data_bin = 2 * val_data_bin - 1
        
    #     # # scaling to -1 -- 1
    #     # pca_realizations = self.min_max_samples(pca_realizations)
    #     # trn_data_pca = self.min_max_samples(trn_data_pca)
    #     # val_data_pca = self.min_max_samples(val_data_pca)
        
    #     trn_model_pca = np.concatenate((trn_data_pca[:self.Nt//5], pca_realizations[:4*self.Nt//5]), axis=0)
    #     Data_trn = np.concatenate((trn_model_pca, trn_data_bin), axis=-1)
    #     Data_val = np.concatenate((val_data_pca, val_data_bin), axis=-1)
    #     Data_trn_paired = Data_val[:self.Nr]
    #     Data_val_paired = Data_val[self.Nr:2*self.Nr]
    #     return Data_trn, Data_trn_paired, Data_val_paired
        
    def build_model(self,
                    model_file=None,
                    summary=False
                    ):
        
        if model_file is not None:
            model_file_dis_pca=model_file+'_pca_discriminator'
            model_file_dis_bin=model_file+'_bin_discriminator'
            model_file_gen_pca=model_file+'_pca_decoder'
            model_file_gen_bin=model_file+'_bin_decoder'
        else:
            model_file_dis_pca=model_file
            model_file_dis_bin=model_file
            model_file_gen_pca=model_file
            model_file_gen_bin=model_file

        # build network models
        K.clear_session()
        self.discriminator_pca = self.nets.build_patch_discriminator(model_shape=(self.n_rows, self.n_cols, 1),
                                                                     filters=self.filters, k_size=self.d_ksize_pca, drop=self.d_drop, 
                                                                     model_file=model_file_dis_pca, summary=summary, name='dis_pca_')
        self.discriminator_bin = self.nets.build_patch_discriminator(model_shape=(self.n_rows, self.n_cols, self.c_dims), drop=self.d_drop,
                                                                     filters=self.filters, k_size=self.d_ksize_bin,
                                                                     model_file=model_file_dis_bin, summary=summary, name='dis_bin_')
        self.decoder_pca = self.nets.build_resnet_generator(model_shape=(self.n_rows, self.n_cols, self.c_dims, 1),
                                                                     filters=self.filters, k_size=self.g_ksize_pca,
                                                                     model_file=model_file_gen_pca, last_act='linear', summary=summary, name='gen_pca_')
        self.decoder_bin = self.nets.build_resnet_generator(model_shape=(self.n_rows, self.n_cols, 1, self.c_dims),
                                                                     filters=self.filters, k_size=self.g_ksize_bin,
                                                                     model_file=model_file_gen_bin, last_act='tanh', summary=summary, name='gen_bin_')
        
        # define input placeholders
        self.real_pca = tf.placeholder(tf.float32,
                                       [self.batch_size, self.n_rows, self.n_cols, 1],
                                        name='real_pca_images')

        self.real_bin = tf.placeholder(tf.float32,
                                       [self.batch_size, self.n_rows, self.n_cols, 2],
                                       name='real_bin_images')
        
        self.real_pca_paired = tf.placeholder(tf.float32,
                                       [self.batch_size, self.n_rows, self.n_cols, 1],
                                        name='real_pca_images_paired')

        self.real_bin_paired = tf.placeholder(tf.float32,
                                       [self.batch_size, self.n_rows, self.n_cols, 2],
                                       name='real_bin_images_paired')
        
        self.fake_pca_paired = self.decoder_pca(self.real_bin_paired)
        self.fake_bin_paired = self.decoder_bin(self.real_pca_paired)
        
        self.fake_pca = self.decoder_pca(self.real_bin)
        self.fake_bin = self.decoder_bin(self.real_pca)

        self.cycle_pca = self.decoder_pca(self.fake_bin)
        self.cycle_bin = self.decoder_bin(self.fake_pca)

        self.D_real_pca, self.D_logits_real_pca = self.discriminator_pca(self.real_pca)
        self.D_fake_pca, self.D_logits_fake_pca = self.discriminator_pca(self.fake_pca)

        self.D_real_bin, self.D_logits_real_bin = self.discriminator_bin(self.real_bin)
        self.D_fake_bin, self.D_logits_fake_bin = self.discriminator_bin(self.fake_bin)
        
        # Reconstruction losses
        self.l1_loss_pca = self.utils.l1_loss(self.real_pca_paired, self.fake_pca_paired)
        self.l1_loss_bin = self.utils.l1_loss(self.real_bin_paired, self.fake_bin_paired)
        
        # cycle losses
        self.cycle_loss_pca = self.utils.cycle_loss(self.real_pca, self.cycle_pca)
        self.cycle_loss_bin = self.utils.cycle_loss(self.real_bin, self.cycle_bin)
        self.total_cycle_loss = self.cycle_loss_pca + self.cycle_loss_bin

        """ Discriminator losses """
        self.d_loss_real_pca = self.utils.cross_entropy_loss(tf.ones_like(self.D_real_pca), self.D_logits_real_pca)
        self.d_loss_fake_pca = self.utils.cross_entropy_loss(tf.zeros_like(self.D_fake_pca), self.D_logits_fake_pca)
        self.d_loss_pca = (self.d_loss_real_pca + self.d_loss_fake_pca) / 2.0
        
        self.d_loss_real_bin = self.utils.cross_entropy_loss(tf.ones_like(self.D_real_bin), self.D_logits_real_bin)
        self.d_loss_fake_bin = self.utils.cross_entropy_loss(tf.zeros_like(self.D_fake_bin), self.D_logits_fake_bin)
        self.d_loss_bin = (self.d_loss_real_bin + self.d_loss_fake_bin) / 2.0
        
        """ Generator losses """
        self.g_loss_pca_ = self.utils.cross_entropy_loss(self.D_logits_fake_pca, tf.ones_like(self.D_logits_fake_pca))
        self.g_loss_bin_ = self.utils.cross_entropy_loss(self.D_logits_fake_bin, tf.ones_like(self.D_logits_fake_bin))
        
        self.g_loss_pca = self.g_loss_pca_ + self.alpha * self.total_cycle_loss + 10 * self.l1_loss_pca
        self.g_loss_bin = self.g_loss_bin_ + self.alpha * self.total_cycle_loss + 10 * self.l1_loss_bin
        
        # self.g_loss_pca = self.g_loss_pca_ + self.alpha * self.total_cycle_loss
        # self.g_loss_bin = self.g_loss_bin_ + self.alpha * self.total_cycle_loss

        t_vars = tf.trainable_variables()
        self.d_vars_pca = [var for var in t_vars if 'dis_pca' in var.name]
        self.d_vars_bin = [var for var in t_vars if 'dis_bin' in var.name]
        self.g_vars_pca = [var for var in t_vars if 'gen_pca' in var.name]
        self.g_vars_bin = [var for var in t_vars if 'gen_bin' in var.name]
        
        self.accuracy = self.utils.accuracy(self.real_bin, self.fake_bin)
        
    def train(self,
              data,
              epochs=1,
              best_acc=-np.inf,
              patience=20,
              split=0.9,
              plots=True,
              num_plots=4,
              plot_freq=2,
              reset_model=True):
        data_trn, data_trn_paired, data_val = self.build_pca_model(data, plots=plots)
        self.fit_gans(data_trn, data_trn_paired, data_val, epochs=epochs, patience=patience, plots=plots, num_plots=num_plots, plot_freq=plot_freq)
    
    def fit_gans(self,
              data_trn,
              data_trn_paired,
              data_val=None,
              epochs=1,
              best_acc=-np.inf,
              patience=20,
              plots=True,
              num_plots=4,
              plot_freq=2,
              reset_model=True):
        """
        optimize the discriminator and generators models using adversarial training
        """
        # Define optimizers
        d_optim_pca = tf.train.AdamOptimizer(self.d_lr_pca, beta1=self.beta1).minimize(self.d_loss_pca, var_list=self.d_vars_pca)
        d_optim_bin = tf.train.AdamOptimizer(self.d_lr_bin, beta1=self.beta1).minimize(self.d_loss_bin, var_list=self.d_vars_bin)
        g_optim_pca = tf.train.AdamOptimizer(self.g_lr_pca, beta1=self.beta1).minimize(self.g_loss_pca, var_list=self.g_vars_pca)
        g_optim_bin = tf.train.AdamOptimizer(self.g_lr_bin, beta1=self.beta1).minimize(self.g_loss_bin, var_list=self.g_vars_bin)

        self.sess = K.get_session()
        init_op = tf.global_variables_initializer()
        if reset_model:            
            self.sess.run(init_op)
            print ("Initializing GANs models variables ...")
        
        print("Starting GANs training ...")
        patience_trn = patience
        start_time = time.time()
        for epoch in range(epochs):
            np.random.shuffle(data_trn)
            np.random.shuffle(data_trn_paired)
            batch_idxs = len(data_trn) // self.batch_size
            lossG_pca, lossG_bin, lossD_pca, lossD_bin, lossC_pca, lossC_bin = [], [], [], [], [], []
            for idx in range(0, batch_idxs):
                batch_files = data_trn[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_files_paired = data_trn_paired[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_images_pca, batch_images_bin = batch_files[:, :, :, :1].astype(np.float32), batch_files[:, :, :, 1:].astype(np.float32)
                batch_images_pca_paired, batch_images_bin_paired = batch_files_paired[:, :, :, :1].astype(np.float32), batch_files_paired[:, :, :, 1:].astype(np.float32)

                # Update D networks
                self.sess.run([d_optim_pca],
                              feed_dict={self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                self.sess.run([d_optim_bin],
                              feed_dict={self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                # Update G networks
                for _ in range(2):
                    self.sess.run([g_optim_pca],
                                 feed_dict={self.real_pca: batch_images_pca, self.real_bin: batch_images_bin,
                                            self.real_pca_paired: batch_images_pca_paired, self.real_bin_paired: batch_images_bin_paired})
                    self.sess.run([g_optim_bin],
                                 feed_dict={self.real_pca: batch_images_pca, self.real_bin: batch_images_bin,
                                            self.real_pca_paired: batch_images_pca_paired, self.real_bin_paired: batch_images_bin_paired})
                    # self.sess.run([g_optim_pca],
                    #               feed_dict={self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                    # self.sess.run([g_optim_bin],
                    #               feed_dict={self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})

                with self.sess.as_default():
                    errD_pca = self.d_loss_pca.eval({self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                    errD_bin = self.d_loss_bin.eval({self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                    errG_pca = self.g_loss_pca_.eval({self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                    errG_bin = self.g_loss_bin_.eval({self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                    errC_pca = self.cycle_loss_pca.eval({self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                    errC_bin = self.cycle_loss_bin.eval({self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                    lossD_pca.append(errD_pca)
                    lossD_bin.append(errD_bin)
                    lossG_pca.append(errG_pca)
                    lossG_bin.append(errG_bin)
                    lossC_pca.append(errC_pca)
                    lossC_bin.append(errC_bin)
                    
                if epoch==0 and idx % (batch_idxs//4)==0 and plots:
                    samples = data_trn[:num_plots]
                    real_samples_pca, real_samples_bin = samples[:, :, :, :1], samples[:, :, :, 1:]
                    fake_pca_img = self.decoder_pca.predict(real_samples_bin)           
                    fake_bin_img = self.decoder_bin.predict(real_samples_pca)
                    fake_bin_img = np.argmax(fake_bin_img, axis=-1)
                    real_samples_bin = np.argmax(real_samples_bin,  axis=-1)
                    PlotDataAE(real_samples_pca[:, :, :, 0], fake_bin_img, digit_size=(self.n_rows, self.n_cols), num=num_plots, Only_Result=True)
                    PlotDataAE(real_samples_bin, fake_pca_img[:, :, :, 0], digit_size=(self.n_rows, self.n_cols), num=num_plots, Only_Result=True)

            if epoch % plot_freq==0 and plots:
                samples = data_trn[:num_plots]
                real_samples_pca, real_samples_bin = samples[:, :, :, :1], samples[:, :, :, 1:]
                fake_pca_img = self.decoder_pca.predict(real_samples_bin)           
                fake_bin_img = self.decoder_bin.predict(real_samples_pca)
                fake_bin_img = np.argmax(fake_bin_img, axis=-1)
                real_samples_bin = np.argmax(real_samples_bin,  axis=-1)
                PlotDataAE(real_samples_pca[:, :, :, 0], fake_bin_img, digit_size=(self.n_rows, self.n_cols), num=num_plots, Only_Result=True)
                PlotDataAE(real_samples_bin, fake_pca_img[:, :, :, 0], digit_size=(self.n_rows, self.n_cols), num=num_plots, Only_Result=True)
                
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_pca: %.8f, d_loss_bin: %.8f, g_loss_pca: %.8f, g_loss_bin: %.8f" \
                        % (epoch+1, idx+1, batch_idxs,
                        time.time() - start_time, np.mean(lossD_pca), np.mean(lossD_bin), np.mean(lossG_pca), np.mean(lossG_bin)))
            print('Cycle losses -->', np.mean(lossC_pca), np.mean(lossC_bin))
            
            with self.sess.as_default():
                batch_idxs = len(data_val) // self.batch_size
                Acc = []
                for idx in range(0, batch_idxs):
                    batch_files = data_val[idx*self.batch_size:(idx+1)*self.batch_size]
                    batch_images = [self.utils.process_data(batch_file) for batch_file in batch_files]
                    batch_images_pca, batch_images_bin = np.array(batch_images)[:, :, :, :1].astype(np.float32), np.array(batch_images)[:, :, :, 1:].astype(np.float32)
                    acc = self.accuracy.eval({self.real_pca: batch_images_pca, self.real_bin: batch_images_bin})
                    Acc.append(acc)
                Acc = np.mean(Acc)
            print("Accuracy:  ", Acc)
            
            if epoch > 5:
                if Acc > best_acc:
                    best_acc = Acc
                    patience_trn = patience
                    if self.saving_path:
                        print('Saving best models ...')
                        Save_Model(self.discriminator_pca, self.saving_path + self.discriminator_name_pca)
                        Save_Model(self.discriminator_bin, self.saving_path + self.discriminator_name_bin)
                        Save_Model(self.decoder_pca, self.saving_path + self.decoder_name_pca)
                        Save_Model(self.decoder_bin, self.saving_path + self.decoder_name_pca)                    
                else:
                    patience_trn -= 1
                if patience_trn < 0:
                    print('Training stoped ...')
                    break
    
    def load_model(self, model_file, summary=True):
        model = self.utils.build_pretrained_model(model_file)
        if summary:
            model.summary()
        return model


""" ////////////////////////////////  3D CLASSES //////////////////////////////////////////// """
class GANS_Wasserstein():
    def __init__(self,latent_dim=500,gen_optimizer = Adam(lr=0.00025, beta_1=0.5),isCVAE=False,
                 dis_optimizer = Adam(lr=5e-5 , beta_1=0.5),save_W=False,printed=10000):
        
        self.img_rows  = 50
        self.img_cols  = 50
        self.img_deep  = 10        
        self.channels  = 3        
        self.img_shape = (self.img_rows, self.img_cols,self.img_deep, self.channels)
        self.latent_dim = latent_dim
        self.n_critic =5
        self.clip_value = 0.01
        self.save=save_W
        self.printed=printed
        
        # Build the generator and discriminator
        if isCVAE:
        	self.generator      = self.build_generatorCVAE()
        else :
        	self.generator      = self.build_generator()
        self.discriminator  = self.build_discriminator()
        
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=dis_optimizer,metrics=['accuracy'])

        self.discriminator.trainable = False        
        self.input_layer = Input(shape=(self.latent_dim,))
        self.generated_volumes = self.generator(self.input_layer)        
        self.generated_volumes_noise  = GaussianNoise(0.01)(self.generated_volumes)
        
        self.validity = self.discriminator(self.generated_volumes_noise)
        self.adversarial_model = Model(inputs=[self.input_layer], outputs=[self.validity])
        self.adversarial_model.summary()
        self.adversarial_model.compile(loss=self.wasserstein_loss,optimizer=gen_optimizer,metrics=['accuracy'])
        
        
        self.x_ =  Lambda(lambda x :  K.argmax(x,axis=-1),output_shape=(50,50,10,))(self.generated_volumes)   
        self.x_0 = Lambda(lambda x : tf.cast(x[:,:,:,:1],tf.float32))(self.x_)  
        self.x_1 = Lambda(lambda x : tf.cast(x[:,:,:,1:2],tf.float32))(self.x_)  
        self.x_2 = Lambda(lambda x : tf.cast(x[:,:,:,2:3],tf.float32))(self.x_)  
        self.x_3 = Lambda(lambda x : tf.cast(x[:,:,:,3:4],tf.float32))(self.x_)  
        self.x_4 = Lambda(lambda x : tf.cast(x[:,:,:,4:5],tf.float32))(self.x_)  
        self.x_5 = Lambda(lambda x : tf.cast(x[:,:,:,5:6],tf.float32))(self.x_)  
        self.x_6 = Lambda(lambda x : tf.cast(x[:,:,:,6:7],tf.float32))(self.x_)  
        self.x_7 = Lambda(lambda x : tf.cast(x[:,:,:,7:8],tf.float32))(self.x_)  
        self.x_8 = Lambda(lambda x : tf.cast(x[:,:,:,8:9],tf.float32))(self.x_)  
        self.x_9 = Lambda(lambda x : tf.cast(x[:,:,:,9:],tf.float32))(self.x_) 

    def _loss_noise(self,x , y):
        """
        Noise reduction
        """
        reconstruction_loss=K.mean(tf.image.total_variation(self.x_0), axis=-1)+K.mean(tf.image.total_variation(self.x_1), axis=-1)+K.mean(tf.image.total_variation(self.x_2), axis=-1)  + K.mean(tf.image.total_variation(self.x_3), axis=-1)+K.mean(tf.image.total_variation(self.x_4), axis=-1)  + K.mean(tf.image.total_variation(self.x_5), axis=-1) +K.mean(tf.image.total_variation(self.x_6), axis=-1)  + K.mean(tf.image.total_variation(self.x_7), axis=-1)  +    K.mean(tf.image.total_variation(self.x_8), axis=-1)  + K.mean(tf.image.total_variation(self.x_9), axis=-1)      
        return reconstruction_loss 

    def wasserstein_loss(self, y_true, y_pred):
        return 1000 * K.mean(y_true * y_pred)
    
    def boundary_loss(self, y_true, y_pred):
        return 50 * K.mean((K.log(y_pred) - K.log(1 - y_pred))**2)
    
    def _loss(self, x , y):
        reconstruction_loss = 100 * binary_crossentropy(x, y)
        return reconstruction_loss 

    def build_generatorCVAE(self):        
        model=Load_Model('/share/GeoFacies/GeoFacies_DL/CVAE3D_New('+str(self.latent_dim)+')_decoder')
        model.summary()
        return model
    
    def build_generator(self):

        model = Sequential()

        model.add(Dense(6* 6* 2 * 64, activation="linear", input_dim=self.latent_dim,kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))        
        model.add(Reshape((6, 6, 2,64)))
        
        model.add(Deconv3D(128, kernel_size=7,strides = 1,padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
                
        model.add(Deconv3D(64, kernel_size=7,strides = (2,2,2),padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Deconv3D(32, kernel_size=(7,7,5),strides = 2,padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Deconv3D(64, kernel_size=(4,4,3),strides = (2,2,1),padding="valid",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))        
        
        
        model.add(Conv3D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
    

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv3D(32, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dropout(rate = 0.75))
        model.add(Conv3D(64, kernel_size=5, strides=2, padding="same"))
        #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dropout(rate = 0.75))
        model.add(Conv3D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dropout(rate = 0.75))
        model.add(Conv3D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dropout(rate = 0.75))
        model.add(Flatten())
        model.add(Dense(1,activation='linear'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train,epochs, batch_size, sample_interval=50,limite=100,std_=1,
              path_save='/share/GeoFacies/temp/ModelGans/Wasserstein _'): 
        self.limite=limite
        valid = -np.reshape( np.ones ((batch_size,)), (-1, 1))
        fake  =  np.reshape( np.ones ((batch_size,)), (-1, 1))   

        iterations = int(2000/batch_size)

        self.history=[]
        self.path_save=path_save
        Dis_loss=[]
        Gen_loss=[]
        for epoch in range(epochs): 
            for _ in range(self.n_critic):                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                noise_real =np.reshape( np.random.uniform(0, 0.01, (batch_size,)), (-1, 1))
                noise_fake =np.reshape( np.random.uniform(0, 0.01, (batch_size,)), (-1, 1))

                noise = np.random.normal(0, std_, (batch_size, self.latent_dim)).astype('float32')
                gen_volumes = self.generator.predict_on_batch(noise)
                self.discriminator.trainable = True    

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                #d_loss = self.discriminator.train_on_batch(np.concatenate((imgs,gen_volumes),axis=0),
                #                                           np.concatenate((valid-noise_real,fake+noise_fake),axis=0))
                loss_real = self.discriminator.train_on_batch(imgs, valid+noise_real)
                loss_fake = self.discriminator.train_on_batch(gen_volumes, fake-noise_fake)
                d_loss = 0.5 * np.add(loss_real, loss_fake)
                Dis_loss.append(d_loss)

                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
                
        
            self.discriminator.trainable = False
            # ---------------------
            #  Train Generator
            # ---------------------
            
            for iii in range(2):
                noise = np.random.normal(0, std_, (batch_size, self.latent_dim)).astype('float32') 
                g_loss = self.adversarial_model.train_on_batch(noise,valid)
                Gen_loss.append(g_loss)


            output="%d [D loss: %f D acc: %f ] [G loss: %f - G acc %f]" % (epoch, d_loss[0],d_loss[-1],g_loss[0], g_loss[-1])
            Printer(output)

            if (epoch % iterations == 0):
                dis_mean_loss=np.array(Dis_loss).mean(axis=0)
                gen_mean_loss=np.array(Gen_loss).mean(axis=0)                
                self.history.append(np.concatenate((dis_mean_loss,gen_mean_loss)))
                Dis_loss=[]
                Gen_loss=[]
                if len(self.history)> 1.5*self.limite:
                    history=np.array(self.history)
                    diff_err=abs(history[:,2]-history[:,0])
                    print("\nDisLoss: ",dis_mean_loss[0], " GenLoss: ",gen_mean_loss[0], " Error : ",diff_err[-1])
                    if self.GansEarlyStopping(diff_err):
                        print("ok")
                        #return
            # If at save interval => save generated image samples
            if (epoch > self.printed) and  (epoch % sample_interval == 0):
                self.sample_images(epoch,std_,save=self.save)                
                
            if g_loss[1] > 1610:
                return

    def GansEarlyStopping(self,loss):
        if loss.shape[0]>self.limite:
            if (loss[-1] < np.min(loss[-self.limite:-1])) and self.save:
                print('save')
                #Save_Model(self.generator,self.path_save+'Best')
                #Save_Model(self.discriminator,self.path_save+'DisBest')

            lastValues=loss[-self.limite:]
            if np.argmin(lastValues)<self.limite/2:
                return True
        return False    

    def ConvertData_Gen(self,x_train):        
        X_train=np.concatenate((x_train[:,:,:,0],x_train[:,:,:,1],x_train[:,:,:,2],x_train[:,:,:,3],x_train[:,:,:,4],
                               x_train[:,:,:,5],x_train[:,:,:,6],x_train[:,:,:,7],x_train[:,:,:,8],x_train[:,:,:,9]),axis=0)
        return X_train     
    
    def PlotModels_Gen(self,X_train):    
        plt.figure(figsize=(20,10))
        for j in range(10):
            plt.subplot(4,10,j+1)
            i=j
            plt.title('Layer_'+str(j+1))
            plt.imshow(X_train[i],cmap="jet")
            plt.axis('off')
        plt.show()
    
    def sample_images(self, epoch,std_,save=False):
        noise = np.random.normal(0, std_, (2, self.latent_dim)).astype('float32')
        gen_imgs = self.generator.predict(noise)
        eval_gen = self.discriminator.predict(gen_imgs)
        generated_volumes=np.argmax(gen_imgs,axis=-1)        
        self.PlotModels_Gen(self.ConvertData_Gen(generated_volumes[0:1,:,:,:]))
        self.PlotModels_Gen(self.ConvertData_Gen(generated_volumes[1:2,:,:,:]))
        #if save :
        #    Save_Model(self.generator,self.path_save+str(epoch))
        #    Save_Model(self.discriminator,self.path_save+'Dis'+str(epoch))
            
    def sample_imagesAll(self, epoch,std_,samples=20,save=False,pt=True):
        noise = np.random.normal(0, std_, (samples, self.latent_dim)).astype('float32')
        gen_imgs = self.generator.predict(noise)
        eval_gen = self.discriminator.predict(gen_imgs)
        index = np.argsort(eval_gen[:,0])
        #generated_volumes=1+gen_imgs[:,:,:,:]
        generated_volumes=np.argmax(gen_imgs,axis=-1)
        print(" Best Values ",  eval_gen[index[-1]] ," and ", eval_gen[index[-2]],' With Mean= ',np.mean(eval_gen))            
        
        if pt:
            for i in range(samples):
                k=-(i+1)
                print("Value :",  eval_gen[index[k]])
                self.PlotModels_Gen(self.ConvertData_Gen(generated_volumes[index[k]:index[k]+1,:,:,:]))

class GANS_Wasserstein_Clear():
    def __init__(self,latent_dim=500,gen_optimizer = Adam(lr=0.00025, beta_1=0.5),isCVAE=False,
                 dis_optimizer = Adam(lr=5e-5 , beta_1=0.5),save_W=False,printed=10000):
        
        self.img_rows  = 50
        self.img_cols  = 50
        self.img_deep  = 10        
        self.channels  = 3        
        self.img_shape = (self.img_rows, self.img_cols,self.img_deep, self.channels)
        self.latent_dim = latent_dim
        self.n_critic =5
        self.clip_value = 0.01
        self.save=save_W
        self.printed=printed
        
        # Build the generator and discriminator
        if isCVAE:
            self.generator      = self.build_generatorCVAE()
        else :
            self.generator      = self.build_generator()
        self.discriminator  = self.build_discriminator()
        
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=dis_optimizer,metrics=['accuracy'])

        self.discriminator.trainable = False        
        self.input_layer = Input(shape=(self.latent_dim,))
        self.generated_volumes = self.generator(self.input_layer)        
        self.generated_volumes_noise  = GaussianNoise(0.01)(self.generated_volumes)
        
        self.validity = self.discriminator(self.generated_volumes_noise)
        self.adversarial_model = Model(inputs=[self.input_layer], outputs=[self.validity])
        self.adversarial_model.summary()
        self.adversarial_model.compile(loss=self.wasserstein_loss,optimizer=gen_optimizer,metrics=['accuracy'])
        
        
        self.x_ =  Lambda(lambda x :  K.argmax(x,axis=-1),output_shape=(50,50,10,))(self.generated_volumes)   
        self.x_0 = Lambda(lambda x : tf.cast(x[:,:,:,:1],tf.float32))(self.x_)  
        self.x_1 = Lambda(lambda x : tf.cast(x[:,:,:,1:2],tf.float32))(self.x_)  
        self.x_2 = Lambda(lambda x : tf.cast(x[:,:,:,2:3],tf.float32))(self.x_)  
        self.x_3 = Lambda(lambda x : tf.cast(x[:,:,:,3:4],tf.float32))(self.x_)  
        self.x_4 = Lambda(lambda x : tf.cast(x[:,:,:,4:5],tf.float32))(self.x_)  
        self.x_5 = Lambda(lambda x : tf.cast(x[:,:,:,5:6],tf.float32))(self.x_)  
        self.x_6 = Lambda(lambda x : tf.cast(x[:,:,:,6:7],tf.float32))(self.x_)  
        self.x_7 = Lambda(lambda x : tf.cast(x[:,:,:,7:8],tf.float32))(self.x_)  
        self.x_8 = Lambda(lambda x : tf.cast(x[:,:,:,8:9],tf.float32))(self.x_)  
        self.x_9 = Lambda(lambda x : tf.cast(x[:,:,:,9:],tf.float32))(self.x_) 

    def _loss_noise(self,x , y):
        """
        Noise reduction
        """
        reconstruction_loss=K.mean(tf.image.total_variation(self.x_0), axis=-1)+K.mean(tf.image.total_variation(self.x_1), axis=-1)+K.mean(tf.image.total_variation(self.x_2), axis=-1)  + K.mean(tf.image.total_variation(self.x_3), axis=-1)+K.mean(tf.image.total_variation(self.x_4), axis=-1)  + K.mean(tf.image.total_variation(self.x_5), axis=-1) +K.mean(tf.image.total_variation(self.x_6), axis=-1)  + K.mean(tf.image.total_variation(self.x_7), axis=-1)  +    K.mean(tf.image.total_variation(self.x_8), axis=-1)  + K.mean(tf.image.total_variation(self.x_9), axis=-1)      
        return reconstruction_loss 

    def wasserstein_loss(self, y_true, y_pred):
        return 1000 * K.mean(y_true * y_pred)
    
    def boundary_loss(self, y_true, y_pred):
        return 50 * K.mean((K.log(y_pred) - K.log(1 - y_pred))**2)
    
    def _loss(self, x , y):
        reconstruction_loss = 100 * binary_crossentropy(x, y)
        return reconstruction_loss 

    def build_generatorCVAE(self):        
        model=Load_Model('/share/GeoFacies/GeoFacies_DL/CVAE3D_New('+str(self.latent_dim)+')_decoder')
        model.summary()
        return model
    
    def build_generator(self):

        model = Sequential()

        model.add(Dense(6* 6* 2 * 64, activation="linear", input_dim=self.latent_dim,kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))        
        model.add(Reshape((6, 6, 2,64)))
        
        model.add(Deconv3D(128, kernel_size=7,strides = 1,padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
                
        model.add(Deconv3D(64, kernel_size=7,strides = (2,2,2),padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Deconv3D(32, kernel_size=(7,7,5),strides = 2,padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Deconv3D(64, kernel_size=(4,4,3),strides = (2,2,1),padding="valid",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))        
        
        
        model.add(Conv3D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
    

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv3D(32, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dropout(rate = 0.75))
        model.add(Conv3D(64, kernel_size=5, strides=2, padding="same"))
        #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dropout(rate = 0.75))
        model.add(Conv3D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dropout(rate = 0.75))
        model.add(Conv3D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
#        model.add(Dropout(rate = 0.75))
        model.add(Flatten())
        model.add(Dense(1,activation='linear'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train,epochs, batch_size, sample_interval=50,limite=100,std_=1,
              path_save='/share/GeoFacies/temp/ModelGans/Wasserstein _'): 
        self.limite=limite
        valid = -np.reshape( np.ones ((batch_size,)), (-1, 1))
        fake  =  np.reshape( np.ones ((batch_size,)), (-1, 1))   

        iterations = int(2000/batch_size)

        self.history=[]
        self.path_save=path_save
        Dis_loss=[]
        Gen_loss=[]
        for epoch in range(epochs): 
            for _ in range(self.n_critic):                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                noise_real =0#np.reshape( np.random.uniform(0, 0.01, (batch_size,)), (-1, 1))
                noise_fake =0#np.reshape( np.random.uniform(0, 0.01, (batch_size,)), (-1, 1))

                noise = np.random.normal(0, std_, (batch_size, self.latent_dim)).astype('float32')
                gen_volumes = self.generator.predict_on_batch(noise)
                self.discriminator.trainable = True    

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                #d_loss = self.discriminator.train_on_batch(np.concatenate((imgs,gen_volumes),axis=0),
                #                                           np.concatenate((valid-noise_real,fake+noise_fake),axis=0))
                loss_real = self.discriminator.train_on_batch(imgs, valid+noise_real)
                loss_fake = self.discriminator.train_on_batch(gen_volumes, fake-noise_fake)
                d_loss = 0.5 * np.add(loss_real, loss_fake)
                Dis_loss.append(d_loss)

                # Clip critic weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
                
        
            self.discriminator.trainable = False
            # ---------------------
            #  Train Generator
            # ---------------------
            
            for iii in range(2):
                noise = np.random.normal(0, std_, (batch_size, self.latent_dim)).astype('float32') 
                g_loss = self.adversarial_model.train_on_batch(noise,valid)
                Gen_loss.append(g_loss)


            output="%d [D loss: %f D acc: %f ] [G loss: %f - G acc %f]" % (epoch, d_loss[0],d_loss[-1],g_loss[0], g_loss[-1])
            Printer(output)

            if (epoch % iterations == 0):
                dis_mean_loss=np.array(Dis_loss).mean(axis=0)
                gen_mean_loss=np.array(Gen_loss).mean(axis=0)                
                self.history.append(np.concatenate((dis_mean_loss,gen_mean_loss)))
                Dis_loss=[]
                Gen_loss=[]
                if len(self.history)> 1.5*self.limite:
                    history=np.array(self.history)
                    diff_err=abs(history[:,2]-history[:,0])
                    print("\nDisLoss: ",dis_mean_loss[0], " GenLoss: ",gen_mean_loss[0], " Error : ",diff_err[-1])
                    if self.GansEarlyStopping(diff_err):
                        return

            # If at save interval => save generated image samples
            if (epoch > self.printed) and  (epoch % sample_interval == 0):
                self.sample_images(epoch,std_,save=self.save)                
                
            if g_loss[1] > 1610:
                return

    def GansEarlyStopping(self,loss):
        if loss.shape[0]>self.limite:
            if (loss[-1] < np.min(loss[-self.limite:-1])) and self.save:
                Save_Model(self.generator,self.path_save+'Best')
                Save_Model(self.discriminator,self.path_save+'DisBest')

            lastValues=loss[-self.limite:]
            if np.argmin(lastValues)<self.limite/2:
                return True
        return False    

    def ConvertData_Gen(self,x_train):        
        X_train=np.concatenate((x_train[:,:,:,0],x_train[:,:,:,1],x_train[:,:,:,2],x_train[:,:,:,3],x_train[:,:,:,4],
                               x_train[:,:,:,5],x_train[:,:,:,6],x_train[:,:,:,7],x_train[:,:,:,8],x_train[:,:,:,9]),axis=0)
        return X_train     
    
    def PlotModels_Gen(self,X_train):    
        plt.figure(figsize=(20,10))
        for j in range(10):
            plt.subplot(4,10,j+1)
            i=j
            plt.title('Layer_'+str(j+1))
            plt.imshow(X_train[i],cmap="jet")
            plt.axis('off')
        plt.show()
    
    def sample_images(self, epoch,std_,save=False):
        noise = np.random.normal(0, std_, (2, self.latent_dim)).astype('float32')
        gen_imgs = self.generator.predict(noise)
        eval_gen = self.discriminator.predict(gen_imgs)
        generated_volumes=np.argmax(gen_imgs,axis=-1)        
        self.PlotModels_Gen(self.ConvertData_Gen(generated_volumes[0:1,:,:,:]))
        self.PlotModels_Gen(self.ConvertData_Gen(generated_volumes[1:2,:,:,:]))
        #if save :
        #    Save_Model(self.generator,self.path_save+str(epoch))
        #    Save_Model(self.discriminator,self.path_save+'Dis'+str(epoch))
            
    def sample_imagesAll(self, epoch,std_,samples=20,save=False,pt=True):
        noise = np.random.normal(0, std_, (samples, self.latent_dim)).astype('float32')
        gen_imgs = self.generator.predict(noise)
        eval_gen = self.discriminator.predict(gen_imgs)
        index = np.argsort(eval_gen[:,0])
        #generated_volumes=1+gen_imgs[:,:,:,:]
        generated_volumes=np.argmax(gen_imgs,axis=-1)
        print(" Best Values ",  eval_gen[index[-1]] ," and ", eval_gen[index[-2]],' With Mean= ',np.mean(eval_gen))            
        
        if pt:
            for i in range(samples):
                k=-(i+1)
                print("Value :",  eval_gen[index[k]])
                self.PlotModels_Gen(self.ConvertData_Gen(generated_volumes[index[k]:index[k]+1,:,:,:]))

class GANS_BASE():
    def __init__(self,latent_dim=100):
        
        self.img_rows  = 50
        self.img_cols  = 50
        self.img_deep  = 10        
        self.channels  = 3        
        self.img_shape = (self.img_rows, self.img_cols,self.img_deep, self.channels)
        self.latent_dim = latent_dim

        gen_optimizer = Adam(lr=0.00025, beta_1=0.5) #,amsgrad=True
        dis_optimizer = Adam(lr=5e-5  , beta_1=0.5)#,amsgrad=True
        # Build the generator and discriminator
        self.generator      = self.build_generator()
        self.discriminator  = self.build_discriminator()
        
        self.discriminator.compile(loss=self._loss, optimizer=dis_optimizer,metrics=['accuracy'])

        self.discriminator.trainable = False        
        self.input_layer = Input(shape=(self.latent_dim,))
        self.generated_volumes = self.generator(self.input_layer)        
        self.generated_volumes_noise  = GaussianNoise(0.01)(self.generated_volumes)

        
        self.validity = self.discriminator(self.generated_volumes_noise)

        self.adversarial_model = Model(inputs=[self.input_layer], outputs=[self.validity])
        self.adversarial_model.summary()
        self.adversarial_model.compile(loss=self._loss,optimizer=gen_optimizer,metrics=['accuracy'])


    def boundary_loss(self, y_true, y_pred):
        return 50 * K.mean((K.log(y_pred) - K.log(1 - y_pred))**2)
    
    def _loss(self, x , y):
        reconstruction_loss = 100 * binary_crossentropy(x, y)
        return reconstruction_loss 

    def _loss_noise(self, x , y):
        reconstruction_loss=K.mean(tf.image.total_variation(x), axis=-1)  
        return reconstruction_loss 
    
    def build_generatorCVAE(self):        
        model=Load_Model('/share/GeoFacies/GeoFacies_DL/CVAE3D_New('+'500'+')_decoder')
        model.summary()
        return model
    
    def build_generator(self):

        model = Sequential()

        model.add(Dense(6* 6* 2 * 64, activation="linear", input_dim=self.latent_dim,kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))        
        model.add(Reshape((6, 6, 2,64)))
        
        model.add(Deconv3D(128, kernel_size=7,strides = 1,padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
                
        model.add(Deconv3D(64, kernel_size=7,strides = (2,2,2),padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Deconv3D(32, kernel_size=(7,7,5),strides = 2,padding="same",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Deconv3D(64, kernel_size=(4,4,3),strides = (2,2,1),padding="valid",kernel_initializer='glorot_normal',bias_initializer='zeros'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))        
        
        
        model.add(Conv3D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
    

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv3D(32, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(64, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train,epochs, batch_size, sample_interval=50,std_=1,
              path_save='/share/GeoFacies/temp/ModelGans/Generator1_'):      
        valid = np.reshape( np.ones((batch_size,)), (-1, 1))
        fake  = np.reshape(np.zeros((batch_size,)), (-1, 1))   
        self.history=[]
        self.path_save=path_save
        for epoch in range(epochs):            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            noise_real =np.reshape( np.random.uniform(0, 0.01, (batch_size,)), (-1, 1))
            noise_fake =np.reshape( np.random.uniform(0, 0.01, (batch_size,)), (-1, 1))


            noise = np.random.normal(0, std_, (batch_size, self.latent_dim)).astype('float32')
            gen_volumes = self.generator.predict_on_batch(noise)
            self.discriminator.trainable = True    

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            #d_loss = self.discriminator.train_on_batch(np.concatenate((imgs,gen_volumes),axis=0),
            #                                           np.concatenate((valid-noise_real,fake+noise_fake),axis=0))
            loss_real = self.discriminator.train_on_batch(imgs, valid)
            loss_fake = self.discriminator.train_on_batch(gen_volumes, fake)

            d_loss = 0.5 * np.add(loss_real, loss_fake)
        
            self.discriminator.trainable = False
            # ---------------------
            #  Train Generator
            # ---------------------
            for iii in range(2):
                noise = np.random.normal(0, std_, (batch_size, self.latent_dim)).astype('float32') 

                g_loss = self.adversarial_model.train_on_batch(noise,valid)
                
            output="%d [D loss: %f D acc: %f ] [G loss: %f - G acc %f]" % (epoch, d_loss[0],d_loss[-1],g_loss[0], g_loss[-1])
            Printer(output)
            self.history.append(np.concatenate((d_loss,g_loss)))
            
            # If at save interval => save generated image samples
            if (epoch > 1000) and  (epoch % sample_interval == 0):
                self.sample_images(epoch,std_,save=False)                
                
            if g_loss[1] > 1610:
                return
    def ConvertData_Gen(self,x_train):
        X_train=np.concatenate((x_train[:,:,:,0],x_train[:,:,:,1],x_train[:,:,:,2],x_train[:,:,:,3],x_train[:,:,:,4],
                               x_train[:,:,:,5],x_train[:,:,:,6],x_train[:,:,:,7],x_train[:,:,:,8],x_train[:,:,:,9]),axis=0)
        return X_train     
    
    def PlotModels_Gen(self,X_train):    
        plt.figure(figsize=(20,10))
        for j in range(10):
            plt.subplot(4,10,j+1)
            i=j
            plt.title('Sampling')
            plt.imshow(X_train[i],cmap="jet")
            plt.axis('off')
        plt.show()
    
    def sample_images(self, epoch,std_,save=False):
        noise = np.random.normal(0, std_, (200, self.latent_dim)).astype('float32')
        gen_imgs = self.generator.predict(noise)
        eval_gen = self.discriminator.predict(gen_imgs)
        index = np.argsort(eval_gen[:,0])
        generated_volumes=np.argmax(gen_imgs,axis=-1)
        if eval_gen[index[-2]] > 0.0:
            print("  Best Values ",  eval_gen[index[-1]] ," and ", eval_gen[index[-2]],' With Mean= ',np.mean(eval_gen))            
            self.PlotModels_Gen(self.ConvertData_Gen(generated_volumes[index[-1]:index[-1]+1,:,:,:]))
            self.PlotModels_Gen(self.ConvertData_Gen(generated_volumes[index[-2]:index[-2]+1,:,:,:]))
            if save :
                Save_Model(self.generator,self.path_save+str(epoch))
                Save_Model(self.discriminator,self.path_save+'Dis'+str(epoch))