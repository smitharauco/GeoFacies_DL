import numpy as np
def CVAE_function(data,dimention_x,dimention_y,comandoEndoder='Encoder',redeVAE='CVAE45(sig)'):
    from keras.models import model_from_json
    from keras.utils import to_categorical
    import keras.backend as K  
    from Model.BiLinearUp import BilinearUpsampling
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
            loaded_model = model_from_json(loaded_model_json,custom_objects={'BilinearUpsampling':BilinearUpsampling })
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

        x_out=x_decoded.reshape((x_decoded.shape[0],dimention_x*dimention_y))
        #Plot_Result(x_decoded,x_decoded)

    #sio.savemat(output,{'Result': x_out})
    K.clear_session()
    return x_out.T
