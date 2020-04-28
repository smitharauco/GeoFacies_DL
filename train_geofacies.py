import argparse
import numpy as np
from pathlib import Path
from Model.DCVAE import DCVAE, DCVAE_Style
from Model.GeoGans import CycleGAN_MPS, GAN2D_MPS, AlphaGAN_MPS, WGAN2D_MPS
from Model.Utils import MPS_Generator
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(description="train Geofacies Class",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="train dataset path (tfrecorfs)")
    parser.add_argument("--test_dataset_path", type=str, default=None,
                        help="test dataset path (tfrecorfs)")
    parser.add_argument("--filters", type=str, default='32-32-32',
                        help="Filters number")
    parser.add_argument("--kernel_dim", type=str, default='3-3-3',
                        help="Dimension of the Kernel")
    parser.add_argument("--strides_values", type=str, default='2-2-2',
                        help="Strides values")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Dimension of the hidden layer")
    parser.add_argument("--latent_dim", type=int, default=500,
                        help="Dimension of the latent vector")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=500,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout rate")                        
    parser.add_argument("--steps", type=int, default=500,
                        help="steps per epoch")
    parser.add_argument("--save_path_weights", type=str, default=None,
                        help="path to save the weights")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--activation", type=str, default="sigmoid",
                        help="activation in the last layer")
    parser.add_argument("--optimizer", type=str, default="RMSprop",
                        help="optimizer ('RMSprop','Adam' or other)")
    parser.add_argument("--model", type=str, default="cvae",
                        help="model architecture ('cvae','cvae-style','0AlphaGAN','CycleGAN','GAN2D_AE' and 'WGAN2D_AE')")
    parser.add_argument("--kl_weight", type=float, default=2.0,
                        help="weight for the KL loss")
    parser.add_argument("--style_weight", type=float, default=3.125e-05,
                        help="weight for the style loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="step patience to stop train")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="epsilon to compute PCA")
    parser.add_argument("--Nr", type=int, default=5000,
                        help="Samples number to compute PCA")
    parser.add_argument("--Nt", type=int, default=5000,
                        help="Number de samples generate by PCA model")  
    parser.add_argument("--alpha", type=int, default=10,
                        help="Hiperparameter of AlphaGans and CycleGans networks")   
    parser.add_argument("--clip", type=float, default=0.05,
                        help="clip value for WGans-AE networks")                                                                                                                    
    args = parser.parse_args()
    return args


def load_data_set(path_tfRecord, isArray=False, batch=4, isTanh=False):
    gen_train = MPS_Generator(path_tfRecord, batch)
    if isArray:
        gen_train = MPS_Generator(path_tfRecord, gen_train.num)
        x_train = gen_train.get_numpy_batch().astype('float32')
        if isTanh:
            x_train = x_train*2-1
    else:
        x_train = gen_train.mps_generator()
    return x_train, gen_train.num, gen_train.image_dim


def load_data_by_class(args,path):
    if path is None:
        return None,None,None
    if args.model == "cvae":
        x_train,nt,image_dim = load_data_set(path,
        batch=args.batch_size)
    elif args.model == "cvae-style" :
        x_train,nt,image_dim = load_data_set(path,isArray=True)
        x_train = np.expand_dims(np.argmax(x_train,axis=-1),axis=-1)
        x_train = x_train*2-1        
    elif args.model == 'AlphaGAN' or args.model == "CycleGAN" or args.model == "GAN2D_AE" or args.model == 'WGAN2D_AE':
        x_train,nt,image_dim= load_data_set(path,
        isArray=True,isTanh=True)
    else:
        print("Don't load dataSet")

    return x_train,nt,image_dim

def main():
    args = get_args()

    kernel = [int(i) for i in args.kernel_dim.split('-')]
    strides = [int(i) for i in args.strides_values.split('-')]
    filters = [int(i) for i in args.filters.split('-')]

    x_train,nt,image_dim = load_data_by_class(args,args.train_dataset_path)
    x_val,vs,_ = load_data_by_class(args,args.test_dataset_path)

    if args.optimizer == 'RMSprop':
        opt = RMSprop(lr=args.lr)
    if args.optimizer == 'Adam':
        opt = Adam(lr=args.lr)

    if args.model == 'AlphaGAN':
        model = AlphaGAN_MPS(input_shape = image_dim,
                d_filters=filters[0], g_filters=filters[0], e_filters=filters[0], c_filters=3500,alpha=args.alpha,
                d_ksize=kernel[0], g_ksize=kernel[0], e_ksize=kernel[0],
                z_size=args.latent_dim, batch_size=args.batch_size,
                saving_path = args.save_path_weights, name='geo_AlphaGAN_', summary=True)
        model.train(x_train, data_val_=x_val, epochs=args.nb_epochs, patience=args.patience, plots=False,reset_model=True) 

    if args.model == 'WGAN2D_AE':
        model = WGAN2D_MPS(input_shape = image_dim,
                d_filters=filters[0], g_filters=filters[0], e_filters=filters[0],clip_value=args.clip,
                d_ksize=kernel[0], g_ksize=kernel[0], e_ksize=kernel[0],
                z_size=args.latent_dim, batch_size=args.batch_size,
                saving_path = args.save_path_weights, name='Wgeo_', summary=True)
        model.train(x_train, data_val=x_val,epochs=args.nb_epochs, patience=args.patience,
         plots=False,reset_model=True)

    if args.model == 'GAN2D_AE' :
        model = GAN2D_MPS(input_shape = image_dim,
                d_filters=filters[0], g_filters=filters[0], e_filters=filters[0],
                d_ksize=kernel[0], g_ksize=kernel[0], e_ksize=kernel[0], z_size=args.latent_dim, batch_size=args.batch_size,
                saving_path = args.save_path_weights, name='geo_', summary=True)
        model.train(x_train,data_val=x_val,epochs=args.nb_epochs, patience=args.patience, plots=False,reset_model=True) 

    if args.model == 'CycleGAN':
        model = CycleGAN_MPS(batch_size=args.batch_size, saving_path = args.save_path_weights, input_shape = image_dim,
        filters=filters[0], epsilon= args.epsilon, Nr=args.Nr, Nt=args.Nt,alpha=args.alpha,
        model_file=None, name='geo_CycleGAN_', summary=True)
        model.train(x_train, epochs=args.nb_epochs, patience=args.patience, plots=False,reset_model=True)    
   
    if args.model == 'cvae-style':
        model = DCVAE_Style(input_shape=x_train.shape[1:],filters=filters,strides=strides,KernelDim=kernel,
        style_weight=args.style_weight,kl_weight=args.kl_weight,act='tanh',
        hidden_dim=args.hidden_dim,latent_dim=args.latent_dim,isTerminal=True,opt=opt,dropout=args.dropout, filepath = args.save_path_weights)
        model.fit(x_train,x_v=x_val,num_epochs=args.nb_epochs, verbose=1,batch_size =  args.batch_size)

    if args.model == 'cvae':
        model = DCVAE(input_shape=image_dim,filters=filters,strides=strides,KernelDim=kernel,
        hidden_dim=args.hidden_dim,latent_dim=args.latent_dim,isTerminal=True,opt=opt,dropout=args.dropout, filepath = args.save_path_weights)
        model.fit_generator(x_train,
                            num_epochs=args.nb_epochs, verbose=1, 
                            steps_per_epoch = nt//args.batch_size,
                            val_set = x_val,
                            validation_steps = vs//args.batch_size)

if __name__ == '__main__':
    main()
