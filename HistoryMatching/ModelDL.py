from Model.BiLinearUp import BilinearUpsampling
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from keras.models import model_from_json
from keras.utils import to_categorical
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
import keras.backend as K


from tensorflow.python.platform import gfile
import keras.backend as K
import tensorflow as tf

def relu6(x):
    return K.relu(x, max_value=6)

class ModelDL_TF: 
    def __init__(self, model, sess):
        self.model = model
        self.sess = sess
        self.name_encoder_in = model['encoder_in']
        self.name_decoder_in = model['decoder_in']
        self.decoder_tensor = self.load_tf(model['path_tf_red_decoder'], 'import/'+model['name_decoder'])
        try:
            self.encoder_tensor = self.load_tf(model['path_tf_red_encoder'], 'import_1/'+model['name_encoder'])
            self.import_name='import_1/'
        except:
            self.encoder_tensor = self.load_tf(model['path_tf_red_encoder'], 'import/'+model['name_encoder'])
            self.import_name='import/'
        self.x, self.y, self.z = self.model['Model_Dim']

    def load_tf(self,path, output_name):
        f = gfile.FastGFile(path, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        f.close()
        self.sess.graph.as_default()
        gf = tf.import_graph_def(graph_def)
        output_tensor = self.sess.graph.get_tensor_by_name(output_name)
        return output_tensor

    def createStateFacies(self, data):

        if self.z > 2:
            data = data.reshape((data.shape[0], ) + (self.x * self.y, self.x), order='F')
        data = data.reshape((data.shape[0], ) + (self.x, self.y, self.z))
        data = to_categorical(data, self.model['NumFacies'])
        if self.model['isTanh']:

            data = 2 * data - 1






        x_encoded = self.sess.run(self.encoder_tensor, {self.import_name+self.name_encoder_in: data})

        return x_encoded

    def updateStateFacies(self, x_encoded):

        x_decoded = self.sess.run(self.decoder_tensor, {'import/'+self.name_decoder_in: x_encoded})
        x_decoded = np.argmax(x_decoded, axis=-1)
        if self.z > 2:
            z_ = x_decoded.reshape((x_decoded.shape[0], self.x * self.y, self.z))
            data = z_.reshape((x_decoded.shape[0], self.x * self.y * self.z), order='F')
        else:
            data = x_decoded.reshape((x_decoded.shape[0], self.x * self.y * self.z))

        return data

    def predict(self, data, is_update=True):
        x_test = data.T.astype('float32')
        x_out = (self.updateStateFacies(x_test) if is_update else self.createStateFacies(x_test))

        return x_out.T


class ModelDL:

    def __init__(self, model, bilinear=True):
        network = model['redePath']

        self.model = model
        self.bilinear = bilinear

        if not (self.model['PCA'] or self.model['IsGans']):
            self.encoder = self.load(network + '_encoder')
        self.decoder = self.load(network + '_decoder')
        self.x, self.y, self.z = self.model['Model_Dim']
        sess = K.get_session()
        self.graph = sess.graph
        #self.graph = tf.Graph()        

    def load(self, model_name):
        model_path = model_name + '.json'
        weights_path = model_name + '_weights.hdf5'

        # load json and create model
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        custom_objects = {'BilinearUpsampling': BilinearUpsampling,'relu6': relu6,'InstanceNormalization': InstanceNormalization } if self.bilinear else {}
        loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)

        # load weights into new model
        try:
            loaded_model.load_weights(weights_path)        
        except:
            if is_gans:
                loaded_model.build((None, self.model['N']))
                loaded_model.load_weights(weights_path)   
        
        return loaded_model

    def createStateFacies(self, data):
        if self.model['PCA']:
            x_test_mean = data - self.model['mu']
            x_encoded = (1/self.model['Z']) * np.dot(self.model['Us_inv'], x_test_mean.T)          
            return x_encoded.T

        if self.model['IsGans']:
            return data

        sess = tf.Session(graph=self.graph)
        if self.z > 2:
            data = data.reshape((data.shape[0], ) + (self.x * self.y, self.z), order='F')

            
        data = data.reshape((data.shape[0], ) + (self.x, self.y, self.z))
        if self.model['toCategorical']:
            data = to_categorical(data, self.model['NumFacies'])

        if self.model['isTanh']:
            data = 2 * data - 1

        x_encoded = self.encoder.predict(data)

        return x_encoded

    def updateStateFacies(self, x_encoded):
        if self.model['PCA']:
            x_encoded = x_encoded.T
            data = self.model['mu'] + (self.model['Z']) * np.dot(self.model['Us'], x_encoded).T 
            if self.z > 2:
                data = data.reshape((data.shape[0], ) + (self.x * self.y, self.z), order='F')            
            x_encoded = data.reshape((data.shape[0], ) + (self.x, self.y, self.z))
                        
        sess = tf.Session(graph=self.graph)
        try :
            x_decoded = self.decoder.predict(x_encoded)
        except :
            x_encoded = np.expand_dims(x_encoded,axis=-1)
            x_decoded = self.decoder.predict(x_encoded)

        if self.model['toCategorical']:
            x_decoded = np.argmax(x_decoded, axis=-1)
        else:
            x_decoded[x_decoded > 0]  = 1
            x_decoded[x_decoded <= 0] = 0

        if self.z > 2:
            z_ = x_decoded.reshape((x_decoded.shape[0], self.x * self.y, self.z))
            data = z_.reshape((x_decoded.shape[0], self.x * self.y * self.z), order='F')
        else:
            data = x_decoded.reshape((x_decoded.shape[0], self.x * self.y * self.z))

        return data

    def predict(self, data, is_update=True):

        x_test = data.T.astype('float32')
        x_out = (self.updateStateFacies(x_test) if is_update else self.createStateFacies(x_test))

        return x_out.T 
