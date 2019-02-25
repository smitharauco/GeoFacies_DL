import numpy as np
import tensorflow as tf
from skimage import transform
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as lays
from Model.Utils import LoadMPS45, LoadMPS100, plot_images, weight_variable, bias_variable, Dense

class DCVAE:
    
    """
    Class to handle building and training CNN-VAE models with tensorflow.
    
    """
    
    def __init__(self, input_shape=(45,45,2), latent_cont_dim=600, opt=tf.train.AdamOptimizer(),
                 hidden_dim=1024, filters=(32,32,16), strides=(2,2,1), dropout=0.1,lr=0.001):
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
        self.model_tf = None
        self.input_shape = input_shape
        self.latent_cont_dim = latent_cont_dim
        self.latent_dim = self.latent_cont_dim
        self.hidden_dim = hidden_dim
        self.filters = filters
        self.strides = strides
        self.dropout = dropout
        self.reshape_decoder = [int(np.ceil(self.input_shape[0]/np.prod(strides))), int(np.ceil(self.input_shape[1]/np.prod(strides))),
                                self.filters[2]]
        self.rate= dropout
        self.lr = lr
        self.sess = tf.Session() 
       
    def _set_model(self):

        """
        Setup model (method should only be called in self.fit())

        """

        # Encoder
        self.inputs = tf.placeholder(shape=(None,)+self.input_shape,dtype=tf.float32)
        self.z_samp = tf.placeholder(shape=(None,self.latent_cont_dim),dtype=tf.float32)            
        self.mean, self.sd    = self.encoder()
        # sampling by re-parameterization technique
        self.epsilon = tf.random_normal(tf.shape(self.mean,out_type = tf.int32), 0, 1, dtype = tf.float32)    
        self.z = tf.add(self.mean ,tf.multiply(self.epsilon, tf.exp(self.sd/2.0)))

        # decoding
        self.decoder_output  =self.decoder(self.z,self.latent_cont_dim,self.hidden_dim,self.reshape_decoder,self.input_shape,reuse=False)
        #y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

        # loss
        out_f = tf.layers.flatten(self.decoder_output) 
        inp_f = tf.layers.flatten(self.inputs)

        # marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)

        encode_decode_loss = tf.keras.losses.binary_crossentropy(inp_f,out_f)
        img_loss =  self.input_shape[0]*self.input_shape[1]*tf.reduce_mean(encode_decode_loss)
        latent_loss = tf.reduce_mean(0.5*tf.reduce_sum(tf.square(self.mean)+ tf.exp(self.sd) -1.0 - self.sd  ,axis=1))
        self.loss = tf.reduce_mean(img_loss + latent_loss)

        self.learning_rate_ = tf.placeholder(shape=[],dtype=tf.float32)
        self.generator(self.z_samp)
        
    # Encoder
    def encoder(self):

        with tf.variable_scope("CVAE_Encoder", reuse=False):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            conv_0 = tf.keras.layers.Conv2D(filters=2,  kernel_size=2, strides=(1, 1), padding='SAME',
                                                 activation='relu')(self.inputs)
            conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1, 1), padding='SAME',
                                                 activation='relu')(conv_0)
            conv_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='SAME',
                                                 activation='relu')(conv_1)
            conv_3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), padding='SAME',
                                                 activation='relu')(conv_2)
            flatten = tf.keras.layers.Flatten()(conv_3)
            
            dense_1 = Dense(flatten, self.reshape_decoder[0]*self.reshape_decoder[1]*self.reshape_decoder[2], self.hidden_dim)
            dropout_1 =  tf.layers.dropout(dense_1, self.rate)
            
            mean = Dense(dropout_1,self.hidden_dim,self.latent_cont_dim)
            sd = Dense(dropout_1,self.hidden_dim,self.latent_cont_dim)

            return mean, sd

        
        # Decoder
    def decoder(self, z, latent_cont_dim, hidden_dim, reshape_decoder,input_shape,reuse=False):
        with tf.variable_scope("CVAE_Decoder", reuse=reuse): 
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

        # 1st hidden desnse  layer

            w0 = tf.get_variable('w0', [z.get_shape()[1], self.hidden_dim], initializer=w_init)
            b0 = tf.get_variable('b0', [self.hidden_dim], initializer=b_init)
            dense_2 = tf.matmul(z, w0) + b0

            #dense_2   = Dense(z,latent_cont_dim,hidden_dim,'Dense_2',reuse=tf.AUTO_REUSE)
            dropout_2 = tf.layers.dropout(dense_2, self.rate)

        # 2do hidden desnse  layer

            w1 = tf.get_variable('w1', [self.hidden_dim, self.reshape_decoder[0]*self.reshape_decoder[1]*self.reshape_decoder[2]], 
                                 initializer=w_init)
            b1 = tf.get_variable('b1', [self.reshape_decoder[0]*self.reshape_decoder[1]*self.reshape_decoder[2]],
                                 initializer=b_init)
            h1 = tf.matmul(dropout_2, w1) + b1

            dense_3   = tf.nn.relu(h1)
            reshape   = tf.keras.layers.Reshape(self.reshape_decoder)(dense_3)

            conv_1_transpose = tf.layers.conv2d_transpose(reshape,filters=16, kernel_size=3,reuse=reuse, 
                                                          strides=(1,1),name='Ct1',activation='relu', padding='SAME')
            conv_2_transpose = tf.layers.conv2d_transpose(conv_1_transpose,filters=32, kernel_size=3,reuse=reuse,name='Ct2', 
                                                          strides=(2,2),activation='relu', padding='SAME')
            conv_3_transpose = tf.layers.conv2d_transpose(conv_2_transpose,filters=32, kernel_size=2,reuse=reuse,name='Ct3', 
                                                          strides=(2,2),activation='relu', padding='SAME')

            bilinear_upsampling_1 = tf.image.resize_bilinear(conv_3_transpose,size=(self.input_shape[1],self.input_shape[1]))
            decoder_output= tf.layers.conv2d(bilinear_upsampling_1,filters=2, kernel_size=2, padding='SAME',activation='sigmoid',
                                             reuse=reuse)
        return decoder_output



    def fit(self, x_train, x_test, epoch=150, batch_size=128, x=10, steph_epoch=5):

        """
        Training model

        """
        self.x = x
        self.steph_epoch = steph_epoch
        self._set_model()
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_).minimize(self.loss)
        self.shape = (batch_size,self.latent_cont_dim)
        
        # Iniciando uma seção e realizando o treinamento 
        #with tf.Session() as sess:
        #    sess.run(tf.global_variables_initializer())   
        self.recon_img_final = []
        self.test_vector = []
        self.train_vector = []
        self.eval_vector = [] 
        self.recon_img = None
        self.sess.run(tf.global_variables_initializer())
        for ep in range(epoch): # Número de épocas de treinamento

            # Treinando o modelo
            train = []
            aux = 0
            for batch_n in range(int(x_train.shape[0]/batch_size)):  # números de batchs
            #for batch_n in range(int(1)):  # números de batchs
                self.batch_img = x_train[aux:aux+batch_size,:,:,:]                     
                _, d = self.sess.run([self.train_op, self.loss], feed_dict={self.inputs: self.batch_img, self.learning_rate_: self.lr})
                aux += batch_size
                train.append(d)
            self.train_vector.append(np.mean(train))
            print('Epoch: {} - Train cost= {:.8f}'.format((ep + 1), np.mean(train)))


            # Testando o treinamento 
            test = []
            aux = 0
            for batch_n in range(int(x_test.shape[0]/batch_size)):
            #for batch_n in range(int(1)):
                self.batch_img = x_test[aux:aux+batch_size,:,:,:]
                self.recon_img, e = self.sess.run([self.decoder_output,self.loss], feed_dict={self.inputs: self.batch_img, self.learning_rate_: self.lr})
                aux += batch_size
                test.append(e)
            self.test_vector.append(np.mean(test))
            print('Epoch: {} - Test cost= {:.8f}'.format((ep + 1), np.mean(test)))  

            self.lr = self.update_lr(self.lr, self.test_vector, x, steph_epoch)
            if self.lr==0:
                break
            #print('Epoca: {} - Learning Rate = {:.8f}'.format((ep + 1),lr))  


            self.new_output = self.sess.run(self.sampler, feed_dict={self.z_samp: np.random.normal(0,1,size=self.shape)})
            #print(self.new_output.max())
            #plot_images(self.new_output, name='Outputs')

    def generator(self,z):
        self.sampler = self.decoder(z, self.latent_cont_dim, self.hidden_dim, self.reshape_decoder, self.input_shape, reuse=True)

    def generator_data(self):
        self.new_output = self.sess.run(self.sampler, feed_dict={self.z_samp: np.random.normal(0,1,size=self.shape)})
        plot_images(self.new_output, name='Outputs')

    # Update Learning Rate
    def update_lr(self,lr, test_vector, x, steph_epoch):
        if len(self.test_vector) > steph_epoch:
            if test_vector[-steph_epoch] < self.test_vector[-1]:
                self.lr = self.lr/x
                if self.test_vector[-2*steph_epoch] < self.test_vector[-1]:
                    return 0      
        return self.lr
   
