import numpy as np
from keras_tqdm import TQDMNotebookCallback
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Conv2D, Conv2DTranspose,Dropout
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
    Class to handle building and training CNN-VAE models.
    """
    def __init__(self, input_shape=(45, 45, 2), latent_cont_dim=512,opt=RMSprop(), hidden_dim=1024, filters=(32,32,16),strides=(2,2,1),dropout=0.1):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        latent_cont_dim : int
            Dimension of continuous latent distribution.

        opt : funtion
            Optimizer to use the network

        hidden_dim : int
            Dimension of hidden layer.

        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of depth.
        
        strides : Array-like, shape (num_strides, num_strides, num_strides)
            Number of strides for each convolution in increasing order of depth.

        dropout : % de dropout [0,1]
        """
        self.opt = opt
        self.model = None
        self.input_shape = input_shape
        self.latent_cont_dim = latent_cont_dim
        self.latent_dim = self.latent_cont_dim
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.strides = strides
        self.dropout = dropout
        self.earlystopper = EarlyStopping(patience=10, verbose=0)
        self.reduce_lr    = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0000005, verbose=1)

    def fit(self, x_train, num_epochs=1, batch_size=100, val_split=.1,reset_model=True,verbose=0):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self._set_model()

        # Update parameters
        #K.set_value(self.opt.lr, learning_rate)
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)

        self.history=self.model.fit(x_train, x_train,
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

        # Instantiate encoder layers
        Q_0 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     activation='relu')
        Q_1 = Conv2D(self.filters[0], (2, 2), padding='same', strides=(self.strides[0], self.strides[0]),
                     activation='relu')
        Q_2 = Conv2D(self.filters[1], (3, 3), padding='same', strides=(self.strides[1],self.strides[1]),
                     activation='relu')
        Q_3 = Conv2D(self.filters[2], (3, 3), padding='same', strides=(self.strides[2], self.strides[2]),
                     activation='relu')
        Q_4 = Flatten()
        Q_5 = Dense(self.hidden_dim, activation='relu')
        Q_6 = Dropout(self.dropout)
        Q_z_mean = Dense(self.latent_cont_dim)
        Q_z_log_var = Dense(self.latent_cont_dim)

        # Set up encoderlatent_disc_dim
        x = Q_0(inputs)
        x = Q_1(x)
        x = Q_2(x)
        x = Q_3(x)
        flat = Q_4(x)
        dp = Q_5(flat)
        hidden= Q_6(dp)

        # Parameters for continous latent distribution
        z_mean = Q_z_mean(hidden)
        z_log_var = Q_z_log_var(hidden)
        self.encoder =Model(inputs, z_mean)

        # Sample from latent distributions
        encoding = Lambda(self._sampling_normal, output_shape=(self.latent_cont_dim,))([z_mean, z_log_var])

        # Generator
        # Instantiate generator layers to be able to sample from latent
        # distribution later
        out_shape = (int(np.ceil(self.input_shape[0] / 4)), int(np.ceil(self.input_shape[1] / 4)), self.filters[2])
        G_0 = Dense(self.hidden_dim, activation='relu')
        G_d = Dropout(self.dropout)
        G_1 = Dense(np.prod(out_shape), activation='relu')
        G_2 = Reshape(out_shape)
        G_3 = Conv2DTranspose(self.filters[2], (3, 3), padding='same',
                              strides=(self.strides[2], self.strides[2]), activation='relu')
        G_4 = Conv2DTranspose(self.filters[1], (3, 3), padding='same',
                              strides=(self.strides[1], self.strides[1]), activation='relu')
        G_5 = Conv2DTranspose(self.filters[0], (2, 2), padding='same',
                              strides=(self.strides[0], self.strides[0]), activation='relu')
        G_5_= BilinearUpsampling(output_size=(self.input_shape[0], self.input_shape[1]))
        G_6 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     strides=(1, 1), activation='sigmoid', name='generated')

        # Apply generator layers
        x = G_0(encoding)
        x = G_d(x)
        x = G_1(x)
        x = G_2(x)
        x = G_3(x)
        x = G_4(x)
        x = G_5(x)
        x = G_5_(x)
        generated = G_6(x)
        self.model =Model(inputs, generated)
        # Set up generator
        inputs_G = Input(batch_shape=(None, self.latent_dim))
        x = G_0(inputs_G)
        x = G_1(x)
        x = G_2(x)
        x = G_3(x)
        x = G_4(x)
        x = G_5(x)
        x = G_5_(x)
        generated_G = G_6(x)
        self.generator = Model(inputs_G, generated_G)

        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        # Compile models
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)
        # Loss and optimizer do not matter here as we do not train these models
        self.generator.compile(optimizer=self.opt, loss='mse')
        self.model.summary()
        print("Completed model setup.")

    def Encoder(self, x_test):
        return self.encoder.predict(x_test)

    def generate(self, number_latent_sample=20,std=1,binary=False):
        """
        Generating examples from samples from the latent distribution.
        """
        latent_sample=np.random.normal(0,std,size=(number_latent_sample,self.latent_cont_dim))
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
        return sampling_normal(z_mean, z_log_var, (None, self.latent_cont_dim))

