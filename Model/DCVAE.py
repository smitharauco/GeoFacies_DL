import numpy as np
from keras_tqdm import TQDMNotebookCallback
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge,Activation
from keras.layers import Conv2D, Conv2DTranspose,Dropout,BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.objectives import binary_crossentropy
from Model.Utils import kl_normal, kl_discrete, sampling_normal,EPSILON
from Model.BiLinearUp import BilinearUpsampling
from keras.callbacks import EarlyStopping,ReduceLROnPlateau

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
        self.reduce_lr    = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0000005, verbose=1)

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
                       callbacks=[TQDMNotebookCallback(),self.earlystopper,self.reduce_lr],
                       validation_split=val_split)
        else:
            self.modelGPU=multi_gpu_model(self.model, gpus=self.multi_GPU)        
            self.modelGPU.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
            self.history=self.modelGPU.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=[TQDMNotebookCallback(),self.earlystopper,self.reduce_lr],
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
#        x = Q_0(inputs)
#        x = Q_1(x)
#        x = Q_2(x)
#        x = Q_3(x)
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
        self.reduce_lr    = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0000005, verbose=1)

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
                       callbacks=[TQDMNotebookCallback(),self.earlystopper,self.reduce_lr],
                       validation_split=val_split)
        else:
            self.modelGPU=multi_gpu_model(self.model, gpus=self.multi_GPU)        
            self.modelGPU.compile(optimizer=self.opt, loss=self._vae_loss,metrics=[self.acc_pred])
            self.history=self.modelGPU.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       verbose=verbose,
                       shuffle=True,
                       callbacks=[TQDMNotebookCallback(),self.earlystopper,self.reduce_lr],
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
#        x = Q_0(inputs)
#        x = Q_1(x)
#        x = Q_2(x)
#        x = Q_3(x)
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
