import pandas as pd
import numpy as np
from scipy import array, linalg, dot
import math
import tensorflow as tf
# Set up eager mode.
#tf.enable_eager_execution()

"""
ES-MDA class implementation with tensorflow
"""
class ESMDA: 
    def __init__(self, N_ens, N_obs, N_m, numsave=2, hasU =False, sess=None, isLoc=False):
        self.N = N_ens
        self.N_m = N_m
        self.N_obs = N_obs
        self.sess = sess
        self.numsave = numsave
        self.decimal = 1-1/math.pow(10, numsave)
        self.isLoc = isLoc
        self.hasU = hasU
        if self.sess is not None:
            self.Create_grahp()

    def Create_grahp(self):
        with tf.device("/cpu:0"):
            self.ph_N = tf.placeholder(tf.float32, name='N')
            self.ph_y_f = tf.placeholder(tf.float32, [self.N_m, self.N], name='y_f')
            self.ph_d_f = tf.placeholder(tf.float32, [self.N_obs, self.N], name='d_f')
            self.ph_Z = tf.placeholder(tf.float32, [self.N_obs, 1], name='Z')
            self.ph_CD = tf.placeholder(tf.float32, [self.N_obs, self.N_obs], name='R')
            self.ph_alpha = tf.placeholder(tf.float32, name='Alpha')
            
            self.corr_t = tf.placeholder(tf.float32, [self.N_m, self.N_obs], name='corr')

            if self.hasU :            
                self.ph_U = tf.placeholder(tf.float32, [self.N_obs, self.N_obs], name='U')
                U_t = self.ph_U
            else:
                R_t = tf.linalg.cholesky(self.ph_CD)
                U_t = tf.transpose(R_t)


            y_m = tf.expand_dims(tf.reduce_mean(self.ph_y_f, axis=1), axis=-1)
            d_m = tf.expand_dims(tf.reduce_mean(self.ph_d_f, axis=1), axis=-1)
            delta_m_f = self.ph_y_f - y_m
            delta_d_f = self.ph_d_f - d_m
            Cdd_ft = tf.matmul(delta_d_f, delta_d_f, transpose_b=True)/(self.ph_N - 1)
            Cmd_ft = tf.matmul(delta_m_f, delta_d_f, transpose_b=True)/(self.ph_N - 1)

            Z_exp = tf.tile(self.ph_Z, tf.constant([1, self.N]))
            mean_t = tf.zeros_like(tf.transpose(Z_exp))



            noise_t = tf.random.normal(mean_t.shape, mean=0.0, stddev=1.0)
            d_obs_t = tf.add(Z_exp, tf.math.sqrt(self.ph_alpha)*tf.matmul(U_t, noise_t, transpose_b=True))

            # Analysis step
            cdd_t = tf.add(Cdd_ft, self.ph_alpha*self.ph_CD)

            fixed_tf_matrix = tf.cast(cdd_t, tf.float64)          
            s_t, u_t, vh_t = tf.linalg.svd(fixed_tf_matrix)   # CPU bether
            v_t = tf.cast(vh_t, tf.float32) 
            s_t = tf.cast(s_t, tf.float32)
            u_t = tf.cast(u_t, tf.float32)

            #s_t, u_t, vh_t = tf.linalg.svd(cdd_t)   # CPU bether
            #v_t = vh_t
            CC = int(self.N_obs*self.decimal)

            zero = tf.constant(0, dtype=tf.float32)
            where = tf.not_equal(s_t, zero)
            index_non_zero = tf.where(where)
            cc_ = tf.shape(tf.boolean_mask(s_t, index_non_zero))[0]

            diagonal_t = s_t[:cc_]
            u_t = u_t[:, :cc_]
            v_t = v_t[:, :cc_]
            s_rt = tf.linalg.diag(tf.math.pow(diagonal_t, -1))
            K_t = tf.matmul(Cmd_ft, (tf.matmul(tf.matmul(v_t, s_rt), u_t, transpose_b= True)))

            if self.isLoc:
                K_t = tf.math.multiply(self.corr_t, K_t)
            self.m_ens_t = self.ph_y_f + tf.matmul(K_t, tf.subtract(d_obs_t, self.ph_d_f)) 

    def Compute(self, m_ens, Z, prod_ens, alpha, CD, corr=[], U=None):
        if not self.isLoc:
            corr = np.zeros([self.N_m, self.N_obs])
        if self.sess is not None:
            if  U is None:
                return self.sess.run(self.m_ens_t, feed_dict={self.ph_alpha: alpha, self.ph_d_f: prod_ens, self.ph_CD: CD, self.ph_y_f:m_ens,self.ph_Z:Z,self.ph_N : self.N,self.corr_t:corr})
            return self.sess.run(self.m_ens_t, feed_dict={self.ph_alpha: alpha, self.ph_d_f: prod_ens, self.ph_CD: CD, self.ph_y_f:m_ens,self.ph_Z:Z,self.ph_N : self.N, self.ph_U: U ,self.corr_t:corr})
        
        return ES_MDA(self.N, m_ens, Z, prod_ens, alpha, CD, corr, self.numsave, U)

def ES_MDA(num_ens,m_ens,Z,prod_ens,alpha,CD,corr,numsave=2):
    varn=1-1/math.pow(10,numsave)
    # Initial Variavel 
    # Forecast step
    yf = m_ens                        # Non linear forward model 
    df = prod_ens                     # Observation Model
    numsave
    ym = np.array(yf.mean(axis=1))    # Mean of the y_f
    dm = np.array(df.mean(axis=1))    # Mean of the d_f
    ym=ym.reshape(ym.shape[0],1)    
    dm=dm.reshape(dm.shape[0],1)    
    dmf = yf - ym
    ddf = df - dm
    
    Cmd_f = (np.dot(dmf,ddf.T))/(num_ens-1);  # The cros-covariance matrix
    Cdd_f = (np.dot(ddf,ddf.T))/(num_ens-1);  # The auto covariance of predicted data
    
    # Perturb the vector of observations
    R = linalg.cholesky(CD,lower=True) #Matriz triangular inferior
    U = R.T   #Matriz R transposta
    p , w =np.linalg.eig(CD)
    
    aux = np.repeat(Z,num_ens,axis=1)
    mean = 0*(Z.T)

    noise=np.random.multivariate_normal(mean[0], np.eye(len(Z)), num_ens).T
    d_obs = aux+math.sqrt(alpha)*np.dot(U,noise)  
    
    # Analysis step
    u, s, vh = linalg.svd(Cdd_f+alpha*CD); v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break
    
    u=u[:,0:i+1]
    v=v[:,0:i+1]
    ss = np.diag(diagonal**(-1))
    K=np.dot(Cmd_f,(np.dot(np.dot(v,ss),(u.T))))
    # Use Kalman covariance
    if len(corr)>0:
        K = corr*K
        
    ya = yf + (np.dot(K,(d_obs-df)))
    m_ens = ya
    return m_ens
