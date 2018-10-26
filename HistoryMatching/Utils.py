from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as sio
import sys
from HistoryMatching.CallNetwork import CVAE_function
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
    try:
        data = sio.loadmat(pathFile)
        var = data[variable]
    except:
        x_Facies = {}
        f = h5py.File(pathFile)
        for k, v in f.items():
            x_Facies[k] = np.array(v)      
        var=x_Facies[variable].astype('float32').T
        f.close()
    
    var = var[:,:num_sampling]        
    return var

def Plot_Result(facies,m_f_a,xp,yp,xi=[],yi=[],initial_data=0,marked='r^',shape=[45,45]):
    plt.rcParams['figure.figsize'] = (15,8)
    fig, axs = plt.subplots(4,5)
    #plt.title(title)
    for i in range(5):
        axs[0,i].imshow(facies[:,i+initial_data].reshape(shape[0],shape[1]), cmap='jet')
        cont = 0
        while cont != len(xp):
            axs[0,i].plot(xp, yp, marked,color='yellow')
            cont +=1
        if len(xi) != 0:
            axs[0,i].plot(xi, yi,'o',color='green')
        axs[0,i].axis('off')
        axs[0,i].set_title('input %i'%(i+1))

        axs[1,i].imshow(m_f_a[:,i+initial_data].reshape(shape[0],shape[1]), cmap='jet')
        cont = 0
        while cont != len(xp):
            axs[1,i].plot(xp, yp, marked,color='yellow')
            cont+=1
        if len(xi) != 0:
            axs[1,i].plot(xi, yi,'o',color='green')
        axs[1,i].axis('off')
        axs[1,i].set_title('output %i'%(i+1))

        axs[2,i].imshow(facies[:,i+5+initial_data].reshape(shape[0],shape[1]), cmap='jet')
        cont = 0
        while cont != len(xp):
            axs[2,i].plot(xp, yp, marked,color='yellow')
            cont += 1 
        if len(xi) != 0:
            axs[2,i].plot(xi, yi,'o',color='green')
        axs[2,i].axis('off')
        axs[2,i].set_title('input %i'%(i+6))

        axs[3,i].imshow(m_f_a[:,i+5+initial_data].reshape(shape[0],shape[1]), cmap='jet')
        cont = 0
        while cont != len(xp):
            axs[3,i].plot(xp, yp, marked,color='yellow')
            cont +=1    
        if len(xi) != 0:
            axs[3,i].plot(xi, yi,'o',color='green')
        axs[3,i].axis('off')
        axs[3,i].set_title('output %i'%(i+6)) 
    plt.show()


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

