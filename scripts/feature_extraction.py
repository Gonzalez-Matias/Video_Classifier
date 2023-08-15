from ast import arg
from email import generator
import tensorflow as tf
from tensorflow import keras
# from keras.applications import VGG16, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from utils.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import argparse
import random
import os
import progressbar
from utils.functions import saliency
import cv2


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('-d', '--dataset', required=True,
    help='path to input dataset')

    ap.add_argument('-o', '--output', required=True,
    help='path to output HDF5 file')

    ap.add_argument('-b', '--batch-size', type=int,
    default=32, help='batch size of images to be passed through network')

    ap.add_argument('-s', '--buffer-size', type=int,
    default=1000, help='size of feature extraction buffer')

    ap.add_argument('-t', '--method', type=str,
    default=None, help='saliency method')

    args = vars(ap.parse_args())
    return args



def get_labels(args, ds_path='data/Videos/train', sc_labels=[], mv_labels=[], n_imgs=0):
    ds_path = args['dataset']
    for root, _, files in os.walk(ds_path):
        for file in files:
            file = file.replace('.mp4','')
            # Since there are cases that mv label is composed by a '_'
            shot_movement = file.split('_')[-1] if ':' in file.split('_')[-1] else file.split('_')[-2]
            shot = shot_movement.split(':')[0]
            movement = shot_movement.split(':')[1]
            sc_labels.append(shot)
            mv_labels.append(movement)
            n_imgs += 1
    return sc_labels, mv_labels, n_imgs


def model_database_prep(args, shotype='sc'):
    print('Loading network...')
    model = keras.applications.VGG16(weights='imagenet', include_top=False)


    sc_labels, mv_labels, ntot_imgs = get_labels(args)
    # init HDF5 dataset writer to store class labels
    dataset = HDF5DatasetWriter((ntot_imgs, 512*7*7), args['output'],
                                dataKey='features', bufSize=args['buffer_size'])
    if shotype == 'sc':
        le = LabelEncoder()
        enc_sc_labels = le.fit_transform(sc_labels)
        # dataset.storeClassLabels(le.classes_)
        return model, dataset, enc_sc_labels, ntot_imgs
    if shotype == 'mv':
        le = LabelEncoder()
        enc_mv_labels = le.fit_transform(mv_labels)
        # dataset.storeClassLabels(le.classes_)
        return model, dataset, enc_mv_labels, ntot_imgs


def extract_process_frame(root, file, method=None, show=False):
    video_full_path = os.path.join(root, file)
    cap = cv2.VideoCapture(video_full_path)
    suc, frame = cap.read()
    image = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    if method is not None:
        image = saliency(image, method=method, show=show)
        # As an output this gives a grayscale(224,224)
        # We will keep it grayscale but we need 3rd dimension for model predict.
        image = np.repeat(image[:,:,np.newaxis] , 3, axis=2)
    else:
        pass
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = keras.applications.imagenet_utils.preprocess_input(image)
    return image


def video_processor(args, shotype='sc',
    input_path='data/Videos/train', batchsize=32,
    i = 0, batch = 0, n_imgs = 0, batchImages = []
    ):

    model, dataset, enc_sc_labels, ntot_imgs = model_database_prep(args, shotype=shotype)

    # init progress bar
    widgets = ['Extracting Features: ', progressbar.Percentage(), ' ',
            progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=ntot_imgs, widgets=widgets).start()

    for root, _, files in os.walk(input_path):
        for file in files:
            image = extract_process_frame(root, file, method=args['method'])
            if i < batchsize:
                batchImages.append(image)
            else:
                batchImages = np.vstack(batchImages)
                features = model.predict(batchImages, batch_size=batchsize)
                features = features.reshape((features.shape[0], 512*7*7))
                dataset.add(features, enc_sc_labels[n_imgs-batchsize:n_imgs])
                batch += 1
                batchImages = []
                i = 0
                batchImages.append(image)
            n_imgs += 1
            i += 1
        pbar.update(n_imgs)

    if batchImages is not None:
        rem_imgs = len(batchImages)
        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size=batchsize)
        features = features.reshape((features.shape[0], 512*7*7))
        dataset.add(features, enc_sc_labels[-rem_imgs:])
        dataset.flush()
    dataset.close()
    pbar.finish()
    print('Features extracted successfuly')
    print(f"Dataset file size is: {round(os.path.getsize(args['output'])*(9.31*10**-10),2)} GB")



def main(args):
    """
    Example of running this script:
    $ python3 scripts/feature_extraction.py --dataset home/data/train --output home/data/features.hdf5

    Note that we can extract features not only from this dataset but also the previous one by Pablo!!!
    We could also use others such as: Animals and CALTECH-101
    """
    # shotype default to 'sc'
    video_processor(args, input_path=args['dataset'],
                    batchsize=args['batch_size'])

if __name__ == '__main__': #__extmain__
    args = parse_args()
    main(args)
