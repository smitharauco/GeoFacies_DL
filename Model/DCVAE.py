import numpy as np
from keras_tqdm import TQDMNotebookCallback
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge,Activation,Add,AveragePooling2D
from keras.layers import Conv2D, Conv2DTranspose,Dropout,BatchNormalization,MaxPooling2D,UpSampling2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import RMSprop,Adam
from keras import backend as K
from keras.objectives import binary_crossentropy
from Model.Utils import kl_normal, kl_discrete, sampling_normal,EPSILON
from Model.BiLinearUp import BilinearUpsampling
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
import math

class DCVAE():
    """
    Class to handle building and training Deep Convolutional Variational Autoencoders models.
    """
    def __init__(self, input_shape=(45,45,2),act='sigmoid', KernelDim=(2,2,3,3),latent_dim=200,opt=RMSprop(),multi_GPU=0,
                 hidden_dim=1024, filters=(2,64, 64, 64),strides=(1,2,1,1),dropout=0):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        latent_dim : int
            Dimension of latent distribution.
            
        opt : Otimazer, method for otimization
        
        hidden_dim : int
            Dimension of hidden layer.

        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of
            depth.
        strides : Array-like, shape (num_filters, num_filters, num_filters)
            Number of strides for each convolution
            
        dropout : % de dropout [0,1]
        
        """
        self.act=act
        self.multi_GPU=multi_GPU
        self.opt = opt
        self.KernelDim=KernelDim
        self.model = None
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.strides = strides
        self.dropout = dropout
        self.earlystopper = EarlyStopping(patience=10, verbose=0)
        self.reduce_lr    = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0000005, verbose=1)
        self.scheduler =True
        self.learningRateScheduler    = LearningRateScheduler(self.step_decay,verbose=0)
        if self.scheduler:
            self.listCall=[self.earlystopper,self.reduce_lr,self.learningRateScheduler, TQDMNotebookCallback()]
        else:
            self.listCall=[self.earlystopper,self.reduce_lr, TQDMNotebookCallback()]

    # learning rate schedule
    def step_decay(self,epoch):
        self.initial_lrate = K.eval(self.model.optimizer.lr)
        drop = 0.8
        epochs_drop = 20.0
        if (1+epoch)%epochs_drop == 0:
            #lrate = self.initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            lrate=self.initial_lrate*drop
        else:
            lrate=self.initial_lrate

        return lrate

    def acc_pred(self,y_true, y_pred):           
        return K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())

        
    def fit(self, x_train, num_epochs=1, batch_size=100, val_split=.1,reset_model=True,verbose=0):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self._set_model()

        # Update parameters
        if self.multi_GPU==0:
            self.model.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
            self.history=self.model.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=self.listCall,
                       validation_split=val_split)
        else:
            self.modelGPU=multi_gpu_model(self.model, gpus=self.multi_GPU)        
            self.modelGPU.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
            self.history=self.modelGPU.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=self.listCall,
                       validation_split=val_split)


    def _set_model(self):
        """
        Setup model (method should only be called in self.fit())
        """
        print("Setting up model...")
        # Encoder
        inputs = Input(batch_shape=(None,) + self.input_shape)
        self.inputs=inputs
        # Instantiate encoder layers        
        for i in range(len(self.filters)):
            if i==0:
                Q = Conv2D(self.filters[i], (self.KernelDim[i], self.KernelDim[i]), 
                           strides=(self.strides[i], self.strides[i]),padding='same',activation='relu')(inputs)
            else:
                Q = Conv2D(self.filters[i], (self.KernelDim[i], self.KernelDim[i]), padding='same',
                                         activation='relu',strides=(self.strides[i], self.strides[i]))(Q)            
                       
        Q_4 = Flatten()
        Q_5 = Dense(self.hidden_dim, activation='relu')
        Q_6 = Dropout(self.dropout)
        Q_z_mean = Dense(self.latent_dim)
        Q_z_log_var = Dense(self.latent_dim)

        # Set up encoder
        flat = Q_4(Q)
        dp = Q_5(flat)
        hidden= Q_6(dp)

        # Parameters for continous latent distribution
        z_mean = Q_z_mean(hidden)
        z_log_var = Q_z_log_var(hidden)
        self.encoder =Model(inputs, z_mean)

        # Sample from latent distributions
        encoding = Lambda(self._sampling_normal, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        self.encoding = encoding
        # Generator
        # Instantiate generator layers to be able to sample from latent
        # distribution later
        out_shape = (int(np.ceil(self.input_shape[0] / np.prod(self.strides) )), int(np.ceil(self.input_shape[1] / np.prod(self.strides))), self.filters[-1])
        
        G_0 = Dense(self.hidden_dim, activation='relu')
        G_d = Dropout(self.dropout)
        G_1 = Dense(np.prod(out_shape), activation='relu')
        G_2 = Reshape(out_shape)
        G=[]
        for i in range(len(self.filters)):
            if i==0:
                G_ = Conv2DTranspose(self.filters[-1], (self.KernelDim[-1], self.KernelDim[-1]), 
                           strides=(self.strides[-1], self.strides[-1]),padding='same',activation='relu')              
            else:
                G_ = Conv2DTranspose(self.filters[-i-1], (self.KernelDim[-i-1], self.KernelDim[-i-1]), padding='same',
                                         activation='relu',strides=(self.strides[-i-1], self.strides[-i-1]))
            G.append(G_)
                
        G_5_= BilinearUpsampling(output_size=(self.input_shape[0], self.input_shape[1]))
        G_6 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     strides=(1, 1), activation=self.act, name='generated')
        # Apply generator layers
        x = G_0(encoding)
        x = G_d(x)
        x = G_1(x)
        x = G_2(x)
        
        for i in range(len(G)):
            x = G[i](x)
            
        x = G_5_(x)
        generated = G_6(x)
        self.model =Model(inputs, generated)
        # Set up generator
        inputs_G = Input(batch_shape=(None, self.latent_dim))
        x = G_0(inputs_G)
        x = G_1(x)
        x = G_2(x)
        
        for i in range(len(self.filters)):
            x = G[i](x)
            
        x = G_5_(x)
        generated_G = G_6(x)
        self.generator = Model(inputs_G, generated_G)

        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        # Compile models
        #self.opt = RMSprop()
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)
        # Loss and optimizer do not matter here as we do not train these models
        self.generator.compile(optimizer=self.opt, loss='mse')
        self.model.summary()
        print("Completed model setup.")

    def Encoder(self, x_test):
        """
        Return predicted result from the encoder model
        """
        return self.encoder.predict(x_test)
    def Decoder(self,x_test,binary=False):
        """
        Return predicted result from the DCVAE model
        """
        if binary:
            return np.argmax(self.model.predict(x_test),axis=-1)
        return self.model.predict(x_test)
        
    def generate(self, number_latent_sample=20,std=1,binary=False):
        """
        Generating examples from samples from the latent distribution.
        """
        latent_sample=np.random.normal(0,std,size=(number_latent_sample,self.latent_dim))
        if binary:
            return np.argmax(self.generator.predict(latent_sample),axis=-1)
        return self.generator.predict(latent_sample)

    def _vae_loss(self, x, x_generated):
        """
        Variational Auto Encoder loss.
        """
        x = K.flatten(x)
        x_generated = K.flatten(x_generated)
        reconstruction_loss = self.input_shape[0] * self.input_shape[1] * \
                                  binary_crossentropy(x, x_generated)
        kl_normal_loss = kl_normal(self.z_mean, self.z_log_var)
        kl_disc_loss = 0
        return reconstruction_loss + kl_normal_loss + kl_disc_loss

    def _sampling_normal(self, args):
        """
        Sampling from a normal distribution.
        """
        z_mean, z_log_var = args
        return sampling_normal(z_mean, z_log_var, (None, self.latent_dim))

class DCVAE_Norm():
    """
    Class to handle building and training Deep Convolutional Variational Autoencoders models.
    """
    def __init__(self, input_shape=(45,45,2),act='sigmoid', KernelDim=(2,2,3,3),latent_dim=200,opt=RMSprop(),multi_GPU=0,
                 hidden_dim=1024, filters=(2,64, 64, 64),strides=(1,2,1,1),dropout=0):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        latent_dim : int
            Dimension of latent distribution.
            
        opt : Otimazer, method for otimization
        
        hidden_dim : int
            Dimension of hidden layer.

        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of
            depth.
        strides : Array-like, shape (num_filters, num_filters, num_filters)
            Number of strides for each convolution
            
        dropout : % de dropout [0,1]
        
        """
        self.act=act
        self.multi_GPU=multi_GPU
        self.opt = opt
        self.KernelDim=KernelDim
        self.model = None
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.strides = strides
        self.dropout = dropout
        self.earlystopper = EarlyStopping(patience=10, verbose=0)
        self.reduce_lr    = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0000005, verbose=1)
        self.scheduler=True        
        self.learningRateScheduler    = LearningRateScheduler(self.step_decay,verbose=0)
        if self.scheduler:
            self.listCall=[self.earlystopper,self.reduce_lr,self.learningRateScheduler, TQDMNotebookCallback()]
        else:
            self.listCall=[self.earlystopper,self.reduce_lr, TQDMNotebookCallback()]

    # learning rate schedule
    def step_decay(self,epoch):
        self.initial_lrate = K.eval(self.model.optimizer.lr)
        drop = 0.8
        epochs_drop = 20.0
        if (1+epoch)%epochs_drop == 0:
            #lrate = self.initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            lrate=self.initial_lrate*drop
        else:
            lrate=self.initial_lrate

        return lrate

    def acc_pred(self,y_true, y_pred):           
        return K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())

        
    def fit(self, x_train, num_epochs=1, batch_size=100, val_split=.1,reset_model=True,verbose=0):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self._set_model()

        # Update parameters
        if self.multi_GPU==0:
            self.model.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
            self.history=self.model.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=self.listCall,
                       validation_split=val_split)
        else:
            self.modelGPU=multi_gpu_model(self.model, gpus=self.multi_GPU)        
            self.modelGPU.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
            self.history=self.modelGPU.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=self.listCall,
                       validation_split=val_split)


    def _set_model(self):
        """
        Setup model (method should only be called in self.fit())
        """
        print("Setting up model...")
        # Encoder
        inputs = Input(batch_shape=(None,) + self.input_shape)
        self.inputs=inputs
        # Instantiate encoder layers        
        for i in range(len(self.filters)):
            if i==0:
                Q = Conv2D(self.filters[i], (self.KernelDim[i], self.KernelDim[i]), 
                           strides=(self.strides[i], self.strides[i]),padding='same')(inputs)
                Q = BatchNormalization()(Q)
                Q = Activation('relu')(Q)
            else:
                Q = Conv2D(self.filters[i], (self.KernelDim[i], self.KernelDim[i]), padding='same',
                                         strides=(self.strides[i], self.strides[i]))(Q)            
                Q = BatchNormalization()(Q)
                Q = Activation('relu')(Q)                
                       
        Q_4 = Flatten()
        Q_5 = Dense(self.hidden_dim)
        Q_50= BatchNormalization()
        Q_51=Activation('relu')
        Q_6 = Dropout(self.dropout)
        Q_z_mean = Dense(self.latent_dim)
        Q_z_log_var = Dense(self.latent_dim)

        # Set up encoder
        flat = Q_4(Q)
        db = Q_5(flat)
        da = Q_50(db)
        dp = Q_51(da)
        hidden= Q_6(dp)

        # Parameters for continous latent distribution
        z_mean = Q_z_mean(hidden)
        z_log_var = Q_z_log_var(hidden)
        self.encoder =Model(inputs, z_mean)

        # Sample from latent distributions
        encoding = Lambda(self._sampling_normal, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        self.encoding = encoding
        # Generator
        # Instantiate generator layers to be able to sample from latent
        # distribution later
        out_shape = (int(np.ceil(self.input_shape[0] / np.prod(self.strides) )), int(np.ceil(self.input_shape[1] / np.prod(self.strides))), self.filters[-1])
        
        G_0 = Dense(self.hidden_dim)
        G_00= BatchNormalization()
        G_01= Activation('relu')
        G_d = Dropout(self.dropout)
        G_1 = Dense(np.prod(out_shape))
        G_10= BatchNormalization()
        G_11= Activation('relu')
        G_2 = Reshape(out_shape)
        G=[]
        for i in range(len(self.filters)):
            if i==0:
                G_ = Conv2DTranspose(self.filters[-1], (self.KernelDim[-1], self.KernelDim[-1]), 
                           strides=(self.strides[-1], self.strides[-1]),padding='same')
                G.append(G_)
                G_ = BatchNormalization()
                G.append(G_)
                G_ = Activation('relu')
                G.append(G_)                
            else:
                G_ = Conv2DTranspose(self.filters[-i-1], (self.KernelDim[-i-1], self.KernelDim[-i-1]), padding='same',
                                         strides=(self.strides[-i-1], self.strides[-i-1]))
                G.append(G_)
                G_ = BatchNormalization()
                G.append(G_)
                G_ = Activation('relu')
                G.append(G_)  
                
        G_5_= BilinearUpsampling(output_size=(self.input_shape[0], self.input_shape[1]))
        G_6 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     strides=(1, 1), activation=self.act, name='generated')
        # Apply generator layers
        x = G_0(encoding)
        x = G_00(x)
        x = G_01(x)
        x = G_d(x)
        x = G_1(x)
        x = G_10(x)        
        x = G_11(x)
        x = G_2(x)
        
        for i in range(len(G)):
            x = G[i](x)
            
        x = G_5_(x)
        generated = G_6(x)
        self.model =Model(inputs, generated)
        # Set up generator
        inputs_G = Input(batch_shape=(None, self.latent_dim))
        x = G_0(inputs_G)
        x = G_00(x)
        x = G_01(x)
        x = G_d(x)        
        x = G_1(x)
        x = G_10(x)        
        x = G_11(x)        
        x = G_2(x)
        
        for i in range(len(G)):
            x = G[i](x)
            
        x = G_5_(x)
        generated_G = G_6(x)
        self.generator = Model(inputs_G, generated_G)

        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        # Compile models
        #self.opt = RMSprop()
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)
        # Loss and optimizer do not matter here as we do not train these models
        self.generator.compile(optimizer=self.opt, loss='mse')
        self.model.summary()
        print("Completed model setup.")

    def Encoder(self, x_test):
        """
        Return predicted result from the encoder model
        """
        return self.encoder.predict(x_test)
    def Decoder(self,x_test,binary=False):
        """
        Return predicted result from the DCVAE model
        """
        if binary:
            return np.argmax(self.model.predict(x_test),axis=-1)
        return self.model.predict(x_test)
        
    def generate(self, number_latent_sample=20,std=1,binary=False):
        """
        Generating examples from samples from the latent distribution.
        """
        latent_sample=np.random.normal(0,std,size=(number_latent_sample,self.latent_dim))
        if binary:
            return np.argmax(self.generator.predict(latent_sample),axis=-1)
        return self.generator.predict(latent_sample)

    def _vae_loss(self, x, x_generated):
        """
        Variational Auto Encoder loss.
        """
        x = K.flatten(x)
        x_generated = K.flatten(x_generated)
        reconstruction_loss = self.input_shape[0] * self.input_shape[1] * \
                                  binary_crossentropy(x, x_generated)
        kl_normal_loss = kl_normal(self.z_mean, self.z_log_var)
        kl_disc_loss = 0
        return reconstruction_loss + kl_normal_loss + kl_disc_loss

    def _sampling_normal(self, args):
        """
        Sampling from a normal distribution.
        """
        z_mean, z_log_var = args
        return sampling_normal(z_mean, z_log_var, (None, self.latent_dim))

class DCVAE_NormV2():
    """
    Class to handle building and training Deep Convolutional Variational Autoencoders models.
    """
    def __init__(self, input_shape=(45,45,2),act='sigmoid', KernelDim=(2,2,3,3),latent_dim=200,opt=RMSprop(),multi_GPU=0,
        filters=(2,64, 64, 64),strides=(1,2,1,1),dropout=0):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        latent_dim : int
            Dimension of latent distribution.
            
        opt : Otimazer, method for otimization
        
        hidden_dim : int
            Dimension of hidden layer.

        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of
            depth.
        strides : Array-like, shape (num_filters, num_filters, num_filters)
            Number of strides for each convolution
            
        dropout : % de dropout [0,1]
        
        """
        self.act=act
        self.multi_GPU=multi_GPU
        self.opt = opt
        self.KernelDim=KernelDim
        self.model = None
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.filters = filters
        self.strides = strides
        self.dropout = dropout
        self.earlystopper = EarlyStopping(patience=10, verbose=0)
        self.reduce_lr    = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0000005, verbose=1)
        self.scheduler=True        
        self.learningRateScheduler    = LearningRateScheduler(self.step_decay,verbose=0)
        if self.scheduler:
            self.listCall=[self.earlystopper,self.reduce_lr,self.learningRateScheduler, TQDMNotebookCallback()]
        else:
            self.listCall=[self.earlystopper,self.reduce_lr, TQDMNotebookCallback()]

    # learning rate schedule
    def step_decay(self,epoch):
        self.initial_lrate = K.eval(self.model.optimizer.lr)
        drop = 0.8
        epochs_drop = 20.0
        if (1+epoch)%epochs_drop == 0:
            #lrate = self.initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            lrate=self.initial_lrate*drop
        else:
            lrate=self.initial_lrate

        return lrate

    def acc_pred(self,y_true, y_pred):           
        return K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())

        
    def fit(self, x_train, num_epochs=1, batch_size=100, val_split=.1,reset_model=True,verbose=0):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self._set_model()

        # Update parameters
        if self.multi_GPU==0:
            self.model.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
            self.history=self.model.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=self.listCall,
                       validation_split=val_split)
        else:
            self.modelGPU=multi_gpu_model(self.model, gpus=self.multi_GPU)        
            self.modelGPU.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
            self.history=self.modelGPU.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=self.listCall,
                       validation_split=val_split)


    def _set_model(self):
        """
        Setup model (method should only be called in self.fit())
        """
        print("Setting up model...")
        # Encoder
        inputs = Input(batch_shape=(None,) + self.input_shape)
        self.inputs=inputs
        # Instantiate encoder layers        
        for i in range(len(self.filters)):
            if i==0:
                Q = Conv2D(self.filters[i], (self.KernelDim[i], self.KernelDim[i]), 
                           strides=(self.strides[i], self.strides[i]),padding='same')(inputs)
                Q = BatchNormalization()(Q)
                Q = Activation('relu')(Q)
            else:
                Q = Conv2D(self.filters[i], (self.KernelDim[i], self.KernelDim[i]), padding='same',
                                         strides=(self.strides[i], self.strides[i]))(Q)            
                Q = BatchNormalization()(Q)
                Q = Activation('relu')(Q)                
                       
        Q_4 = Flatten()
        Q_6 = Dropout(self.dropout)
        Q_z_mean = Dense(self.latent_dim)
        Q_z_log_var = Dense(self.latent_dim)

        # Set up encoder
        flat = Q_4(Q)
        hidden= Q_6(flat)

        # Parameters for continous latent distribution
        z_mean = Q_z_mean(hidden)
        z_log_var = Q_z_log_var(hidden)
        self.encoder =Model(inputs, z_mean)

        # Sample from latent distributions
        encoding = Lambda(self._sampling_normal, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        self.encoding = encoding
        # Generator
        # Instantiate generator layers to be able to sample from latent
        # distribution later
        out_shape = (int(np.ceil(self.input_shape[0] / np.prod(self.strides) )), int(np.ceil(self.input_shape[1] / np.prod(self.strides))), self.filters[-1])
        
        #G_0 = Dense(self.hidden_dim)
        #G_00= BatchNormalization()
        #G_01= Activation('relu')
        G_d = Dropout(self.dropout)
        G_1 = Dense(np.prod(out_shape))
        G_10= BatchNormalization()
        G_11= Activation('relu')
        G_2 = Reshape(out_shape)
        G=[]
        for i in range(len(self.filters)):
            if i==0:
                G_ = Conv2DTranspose(self.filters[-1], (self.KernelDim[-1], self.KernelDim[-1]), 
                           strides=(self.strides[-1], self.strides[-1]),padding='same')
                G.append(G_)
                G_ = BatchNormalization()
                G.append(G_)
                G_ = Activation('relu')
                G.append(G_)                
            else:
                G_ = Conv2DTranspose(self.filters[-i-1], (self.KernelDim[-i-1], self.KernelDim[-i-1]), padding='same',
                                         strides=(self.strides[-i-1], self.strides[-i-1]))
                G.append(G_)
                G_ = BatchNormalization()
                G.append(G_)
                G_ = Activation('relu')
                G.append(G_)  
                
        G_5_= BilinearUpsampling(output_size=(self.input_shape[0], self.input_shape[1]))
        G_6 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     strides=(1, 1), activation=self.act, name='generated')
        # Apply generator layers
        #x = G_0(encoding)
        #x = G_00(x)
        #x = G_01(x)
        x = G_d(encoding)
        x = G_1(x)
        x = G_10(x)        
        x = G_11(x)
        x = G_2(x)
        
        for i in range(len(G)):
            x = G[i](x)
            
        x = G_5_(x)
        generated = G_6(x)
        self.model =Model(inputs, generated)
        # Set up generator
        inputs_G = Input(batch_shape=(None, self.latent_dim))
        #x = G_0(inputs_G)
        #x = G_00(x)
        #x = G_01(x)
        x = G_d(inputs_G)        
        x = G_1(x)
        x = G_10(x)        
        x = G_11(x)        
        x = G_2(x)
        
        for i in range(len(G)):
            x = G[i](x)
            
        x = G_5_(x)
        generated_G = G_6(x)
        self.generator = Model(inputs_G, generated_G)

        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        # Compile models
        #self.opt = RMSprop()
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)
        # Loss and optimizer do not matter here as we do not train these models
        self.generator.compile(optimizer=self.opt, loss='mse')
        self.model.summary()
        print("Completed model setup.")

    def Encoder(self, x_test):
        """
        Return predicted result from the encoder model
        """
        return self.encoder.predict(x_test)
    def Decoder(self,x_test,binary=False):
        """
        Return predicted result from the DCVAE model
        """
        if binary:
            return np.argmax(self.model.predict(x_test),axis=-1)
        return self.model.predict(x_test)
        
    def generate(self, number_latent_sample=20,std=1,binary=False):
        """
        Generating examples from samples from the latent distribution.
        """
        latent_sample=np.random.normal(0,std,size=(number_latent_sample,self.latent_dim))
        if binary:
            return np.argmax(self.generator.predict(latent_sample),axis=-1)
        return self.generator.predict(latent_sample)

    def _vae_loss(self, x, x_generated):
        """
        Variational Auto Encoder loss.
        """
        x = K.flatten(x)
        x_generated = K.flatten(x_generated)
        reconstruction_loss = self.input_shape[0] * self.input_shape[1] * \
                                  binary_crossentropy(x, x_generated)
        kl_normal_loss = kl_normal(self.z_mean, self.z_log_var)
        kl_disc_loss = 0
        return reconstruction_loss + kl_normal_loss + kl_disc_loss

    def _sampling_normal(self, args):
        """
        Sampling from a normal distribution.
        """
        z_mean, z_log_var = args
        return sampling_normal(z_mean, z_log_var, (None, self.latent_dim))

class DCVAE_Inc():
    """
    Class to handle building and training Inception-VAE models.
    """
    def __init__(self, input_shape=(100, 100, 2), latent_dim=100,kernel_init=32,opt=Adam(amsgrad=True),drop=0.1,scheduler=False):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        latent_dim : int
            Dimension of latent distribution.

        kernel_init : int
            Dimension of kernel.

        opt : Otimazer

        """
        self.initial_lrate=0.001
        self.drop = drop
        self.opt = opt
        self.model = None
        self.generator = None
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.kernel_init = kernel_init
        self.earlystopper = EarlyStopping(patience=10, verbose=0)
        self.reduce_lr    = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0000005, verbose=1)
        self.scheduler =scheduler
        self.learningRateScheduler    = LearningRateScheduler(self.step_decay,verbose=0)
        if self.scheduler:
            self.listCall=[self.earlystopper,self.reduce_lr,self.learningRateScheduler, TQDMNotebookCallback()]
        else:
            self.listCall=[self.earlystopper,self.reduce_lr, TQDMNotebookCallback()]

    # learning rate schedule
    def step_decay(self,epoch):
        self.initial_lrate = K.eval(self.model.optimizer.lr)
        drop = 0.8
        epochs_drop = 20.0
        if (1+epoch)%epochs_drop == 0:
            #lrate = self.initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            lrate=self.initial_lrate*drop
        else:
            lrate=self.initial_lrate

        return lrate


    def acc_pred(self,y_true, y_pred):
        return K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx())

    def fit(self, x_train, num_epochs=1, batch_size=100, val_split=.1,reset_model=True,verbose=0):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self._set_model()

        # Update parameters

        #self.model.compile(optimizer=Adam(amsgrad=True), loss=self._vae_loss,metrics=[self.acc_pred])
        #self.model.fit(x_train, x_train,epochs=5,batch_size=self.batch_size,verbose=1,validation_split=val_split)

        self.model.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
        self.history=self.model.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=self.listCall,
                       validation_split=val_split)

    def create_encoder_single_conv(self,in_chs, out_chs, kernel):
        assert kernel % 2 == 1
        model=Conv2D(out_chs, kernel_size=kernel, padding='same')(in_chs)
        model=BatchNormalization()(model)
        model=Activation('relu')(model)
        return model

    ## Encoder Inception Signle
    def create_encoder_inception_signle(self,in_chs, out_chs):
        channels = out_chs
        bn_ch = channels // 2
        bottleneck = self.create_encoder_single_conv(in_chs, bn_ch, 1)    
        conv1 = self.create_encoder_single_conv(bottleneck, channels, 1)
        conv3 = self.create_encoder_single_conv(bottleneck, channels, 3)
        conv5 = self.create_encoder_single_conv(bottleneck, channels, 5)
        conv7 = self.create_encoder_single_conv(bottleneck, channels, 7)
        pool3 = MaxPooling2D(3,strides=1,padding='SAME')(in_chs)
        pool5 = MaxPooling2D(5,strides=1,padding='SAME')(in_chs)
        return Add()([conv1,conv3,conv5,conv7,pool3,pool5])

    def create_downsampling_module(self,input_, pooling_kenel,filters):    
        av1  = AveragePooling2D(pool_size=pooling_kenel)(input_)
        cv1  = self.create_encoder_single_conv(av1,filters,1)   
        return cv1

    def create_decoder_single_conv(self,in_chs, out_chs, kernel,stride=1):
        model=Conv2DTranspose(out_chs, kernel_size=kernel,strides=stride, padding='same')(in_chs)
        model=BatchNormalization()(model)
        model=Activation('relu')(model)
        return model

    ## Decoder Inception Signle
    def create_decoder_inception_signle(self,in_chs, out_chs):
        channels = out_chs
        bn_ch = channels // 2
        bottleneck = self.create_decoder_single_conv(in_chs, bn_ch, 1)    
        conv1 = self.create_decoder_single_conv(bottleneck, channels, 1)
        conv3 = self.create_decoder_single_conv(bottleneck, channels, 3)
        conv5 = self.create_decoder_single_conv(bottleneck, channels, 5)
        conv7 = self.create_decoder_single_conv(bottleneck, channels, 7)
        pool3 = MaxPooling2D(3,strides=1,padding='SAME')(in_chs)
        pool5 = MaxPooling2D(5,strides=1,padding='SAME')(in_chs)
        return Add()([conv1,conv3,conv5,conv7,pool3,pool5])

    def create_upsampling_module(self,input_, pooling_kenel,filters):    
        cv1  = self.create_decoder_single_conv(input_,filters,pooling_kenel,stride=pooling_kenel)   
        return cv1

    def createEncoder(self,input_layer):
        upch1  = Conv2D(self.kernel_init, padding='same', kernel_size=1)(input_layer)
        stage1 = self.create_encoder_inception_signle(upch1,self.kernel_init)
        upch2  = self.create_downsampling_module(stage1,2,self.kernel_init*2)

        stage2 = self.create_encoder_inception_signle(upch2,self.kernel_init*2)
        upch3  = self.create_downsampling_module(stage2,2,self.kernel_init*4) 

        stage3 = self.create_encoder_inception_signle(upch3,self.kernel_init*4)
        upch4  = self.create_downsampling_module(stage3,4,self.kernel_init*8) 

        stage4 = self.create_encoder_inception_signle(upch4,self.kernel_init*8)
        out    = AveragePooling2D(self.input_shape[0]//16)(stage4)

        sq1 = Lambda(lambda x: K.squeeze(x, -2))(out)
        sq2 = Lambda(lambda x: K.squeeze(x, -2))(sq1)
        return sq2


        up1 = UpSampling3D((int(np.ceil(self.input_shape[0]/16))-1,int(np.ceil(self.input_shape[0]/16))-1,1))(sq3)

    def createDecoder(self,input_):

        sq1 = Lambda(lambda x: K.expand_dims(x, 1))(input_)
        sq2 = Lambda(lambda x: K.expand_dims(x, 1))(sq1)
        up1 = UpSampling2D(int(np.ceil(self.input_shape[0]/16)))(sq2)

        stage1 = self.create_decoder_inception_signle(up1,self.kernel_init*8)

        downch1 = self.create_upsampling_module(stage1,4,self.kernel_init*4)

        stage2  = self.create_decoder_inception_signle(downch1,self.kernel_init*4)
        downch2 = self.create_upsampling_module(stage2,2,self.kernel_init*2)

        stage3 = self.create_decoder_inception_signle(downch2,self.kernel_init*2)
        downch3 = self.create_upsampling_module(stage3,2,self.kernel_init)

        stage4 = self.create_decoder_inception_signle(downch3,self.kernel_init)
        stage5 = BilinearUpsampling(output_size=(self.input_shape[0], self.input_shape[1]))(stage4)
        last = Conv2DTranspose(self.input_shape[-1], kernel_size=1,activation='sigmoid')(stage5)
        return last


    def _set_model(self):
        """
        Setup model (method should only be called in self.fit())
        """
        print("Setting up model...")
        # Encoder
        inputs = Input(batch_shape=(None,) + self.input_shape)

        baseEncoder = self.createEncoder(inputs)
        baseEncoder = Dropout(self.drop)(baseEncoder)

        # Instantiate encoder layers
        Q_z_mean = Dense(self.latent_dim)
        Q_z_log_var = Dense(self.latent_dim)

        # Parameters for continous latent distribution
        z_mean = Q_z_mean(baseEncoder)
        z_log_var = Q_z_log_var(baseEncoder)
        self.encoder =Model(inputs, z_mean)

        # Sample from latent distributions

        encoding = Lambda(self._sampling_normal, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        G_0 = Dense(8*self.kernel_init)(encoding)
        G_0 = Dropout(self.drop)(G_0)
        baseDecoder = self.createDecoder(G_0)

        self.model =Model(inputs, baseDecoder)
        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var


        # Compile models
        #self.opt = RMSprop()
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)
        self.model.summary()
        print("Completed model setup.")
    def Encoder(self, x_test):
        return self.encoder.predict(x_test)

    def generate(self, number_latent_sample=20,std=1,binary=False):
        """
        Generating examples from samples from the latent distribution.
        """
        if self.generator==None:
            # Set up generator
            inputs_G = Input(batch_shape=(None, self.latent_dim))
            G_0 = Dense(8*self.kernel_init)(inputs_G)
            G_0 = Dropout(self.drop)(G_0)
            generated = self.createDecoder(G_0)
            self.generator =Model(inputs_G, generated)
            for i,l in enumerate(self.model.layers):       
                if i > 92:        
                    self.generator.layers[i-92].set_weights(self.model.layers[i].get_weights())

            # Loss and optimizer do not matter here as we do not train these models
            self.generator.compile(optimizer=self.opt, loss='mse')

        latent_sample=np.random.normal(0,std,size=(number_latent_sample,self.latent_dim))
        if binary:
            return np.argmax(self.generator.predict(latent_sample),axis=-1)
        return self.generator.predict(latent_sample)

    def _vae_loss(self, x, x_generated):
        """
        Variational Auto Encoder loss.
        """
        x = K.flatten(x)
        x_generated = K.flatten(x_generated)
        reconstruction_loss = self.input_shape[0] * self.input_shape[1] * \
                                  binary_crossentropy(x, x_generated)
        kl_normal_loss = kl_normal(self.z_mean, self.z_log_var)

        return reconstruction_loss + kl_normal_loss

    def _sampling_normal(self, args):
        """
        Sampling from a normal distribution.
        """
        z_mean, z_log_var = args
        return sampling_normal(z_mean, z_log_var, (None, self.latent_dim))

    def _sampling_concrete(self, args):
        """
        Sampling from a concrete distribution
        """
        return sampling_concrete(args, (None, self.latent_disc_dim))
