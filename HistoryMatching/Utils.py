from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.45
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import model_from_json
import scipy.io as sio
import sys
from HistoryMatching.ES_MDA import ES_MDA
import matplotlib.pyplot as plt


def GerenateObsFile(position,Facies,filePath='Obs.dat',dim_shape=[45,45]):
    #pathFile='Ensemble.mat'
    #data = sio.loadmat(pathFile)
    #var = data['Dato']
    #Facies = var[:,5999]
    obs=GetFaciesData(Facies,dim_shape,position,filePath)
    return obs

def Load_Ensemble(pathFile='Ensemble.mat',variable='Dato',num_sampling=200):
    data = sio.loadmat(pathFile)
    var = data[variable]
    var = var[:,:num_sampling]
    return var

def Plot_Result(facies,m_f_a,xp,yp,xi=[],yi=[],initial_data=0,marked='r^'):
    plt.rcParams['figure.figsize'] = (15,8)
    fig, axs = plt.subplots(4,5)
    #plt.title(title)
    for i in range(5):
        axs[0,i].imshow(facies[:,i+initial_data].reshape(45,45), cmap='jet')
        cont = 0
        while cont != len(xp):
            axs[0,i].plot(xp, yp, marked,color='yellow')
            cont +=1
        if len(xi) != 0:
            axs[0,i].plot(xi, yi,'o',color='green')
        axs[0,i].axis('off')
        axs[0,i].set_title('input %i'%(i+1))

        axs[1,i].imshow(m_f_a[:,i+initial_data].reshape(45,45), cmap='jet')
        cont = 0
        while cont != len(xp):
            axs[1,i].plot(xp, yp, marked,color='yellow')
            cont+=1
        if len(xi) != 0:
            axs[1,i].plot(xi, yi,'o',color='green')
        axs[1,i].axis('off')
        axs[1,i].set_title('output %i'%(i+1))

        axs[2,i].imshow(facies[:,i+5+initial_data].reshape(45,45), cmap='jet')
        cont = 0
        while cont != len(xp):
            axs[2,i].plot(xp, yp, marked,color='yellow')
            cont += 1 
        if len(xi) != 0:
            axs[2,i].plot(xi, yi,'o',color='green')
        axs[2,i].axis('off')
        axs[2,i].set_title('input %i'%(i+6))

        axs[3,i].imshow(m_f_a[:,i+5+initial_data].reshape(45,45), cmap='jet')
        cont = 0
        while cont != len(xp):
            axs[3,i].plot(xp, yp, marked,color='yellow')
            cont +=1    
        if len(xi) != 0:
            axs[3,i].plot(xi, yi,'o',color='green')
        axs[3,i].axis('off')
        axs[3,i].set_title('output %i'%(i+6)) 
    plt.show()


def CVAE_function(data,dimention_x,dimention_y,comandoEndoder='Encoder',redeVAE='CVAE45(sig)'):
    
    #redeVAE="S:\ModelosDL\MPS45\CVAE45(sig)"
    function=comandoEndoder
    # Arquivo mat input
    # arquivo salida
    #output="S:\ModelosDL\MPS45\EncoderOut.mat"

    def load_AE(name):
        def load(model_name):
            model_path = "%s.json" % model_name
            weights_path = "%s_weights.hdf5" % model_name
            # load json and create model
            json_file = open(model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()      
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(weights_path)        
            return loaded_model
        encoder=load(name+"_encoder")
        Decoder=load(name+"_decoder")
        return encoder,Decoder


    encoder,decoder=load_AE(redeVAE)
    #EnsIni= sio.loadmat(a_mat)
    if function=="Encoder":
        #x_test=(EnsIni['Facie']).astype('float32')
        x_test=data.T
        x_test=x_test.reshape((x_test.shape[0],) + (dimention_x,dimention_y,1))
        #x_test=x_test*2-1
        x_test=to_categorical(x_test,2)
        x_out = encoder.predict(x_test)
        #Plot_Result(x_test,x_test)
    if function=="Decoder":
        x_test=data.T
        x_decoded = decoder.predict(x_test)
        x_decoded=np.argmax(x_decoded,axis=-1)
        #Plot_Result(x_decoded,x_decoded)

        x_out=x_decoded.reshape((x_decoded.shape[0],45*45))
        #Plot_Result(x_decoded,x_decoded)

    #sio.savemat(output,{'Result': x_out})
    return x_out.T

def CreateStateFacies(data,dimention_x,dimention_y,redeVAE='CVAE45(sig)'):
    M_Rep = CVAE_function(data,dimention_x,dimention_y,'Encoder',redeVAE)
    return M_Rep

def UpdateStateFacies(data,dimention_x,dimention_y,redeVAE='CVAE45(sig)'):
    M_Face = CVAE_function(data,dimention_x,dimention_y,'Decoder',redeVAE)
    return M_Face

def GetFaciesData(facies,dimention,position,path_save):
    layers = facies.reshape([int(len(facies)/dimention[1]),dimention[1]])
    std = 0.1    
    values = [layers[position[i,0],position[i,1]] for i  in range(position.shape[0])]
    if (path_save==''):
        return values
    file = open(path_save,'w') 
    file.write('TIME 1 1\n') 
    for i in values:
        file.write('1 '+str(i)+' '+str(std)+'\n')
    file.close()
    return values

def Contitional_ES_MDA(alp,Corr,position,obs,R,m_x,m_f,dim_shape,redeVAE):
    Alpha = np.ones((alp),dtype=int)*alp
    for t in range(len(Alpha)):
        Obs_sim = [((GetFaciesData(m_f[:,i],dim_shape,position,''))) for i in range(m_f.shape[1])]
        Obs_sim=  np.array(Obs_sim).T
        print('Erro ite_',t, ' : ' ,sum(sum(abs(Obs_sim-obs))))    
        m_x = ES_MDA(m_f.shape[1],m_x,obs,Obs_sim,Alpha[t],R,Corr,2)
        m_f = UpdateStateFacies(m_x,dim_shape[0],dim_shape[1],redeVAE)
    Obs_sim = [((GetFaciesData(m_f[:,i],dim_shape,position,''))) for i in range(m_f.shape[1])]
    Obs_sim=  np.array(Obs_sim).T
    print('Erro End: ',sum(sum(abs(Obs_sim-obs))))
    return m_f

