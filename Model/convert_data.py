import argparse
import numpy as np
from Utils import convert_to
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# cd 'GeoFacies/GeoFaciesICA/GeoFacies_DL/Model'
# python convert_data.py --dataset_path_input '/share/GeoFacies/DataSet/MPS45/MPS45.npy' --dataset_path_output '../../../Maykol/dataset/MPS45' --split

def get_args():

    parser = argparse.ArgumentParser(description="Convert to tfrecords from numpy - Geofacies",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_path_input", type=str, required=True, default = "",
                        help="dataset path (numpy)")
    parser.add_argument("--dataset_path_output", type=str, required=True, default = "",
                        help="dataset path (tfrecorfs)")
    parser.add_argument("--split", action='store_true',
                        help="train-test split")                  
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="test size of split")
    parser.add_argument("--random_state", type=int, default=0,
                        help="random seed")
    parser.add_argument("--model", type=str, default="cvae",
                        help="model architecture ('cvae','cvae_style','CycleGAN')")
    args = parser.parse_args()
    
    return args


def main():
    args = get_args()

    data = np.load(args.dataset_path_input)
    data = to_categorical(data,2)

    if args.model == "cvae_style":
        data = np.argmax(data,axis=-1)
        data = np.expand_dims(data, axis=-1)
        data = 2*data-1

    if args.split:
        x_train, x_test  = train_test_split(data, test_size=args.test_size, random_state=args.random_state)
        convert_to(x_test, 'test_val', args.dataset_path_output)
    else:
        x_train = data

    convert_to(x_train, 'train', args.dataset_path_output)

if __name__ == '__main__':
    main()