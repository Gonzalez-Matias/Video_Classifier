from logging import raiseExceptions
import boto3
import os
import argparse
from random import randint
import cv2
import numpy as np

def download_data(quantity='all'):
    # fetch credentials from env variablesenv
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    # setup a AWS S3 client/resource
    s3 = boto3.resource(
        's3', 
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        )
    # point the resource at the existing bucket
    bucket = s3.Bucket('anyoneai-datasets')
    if quantity == 'all':
        for file in bucket.objects.filter(Prefix = 'shot-type/movie-shot-trailers'):
            # ['shot-type', 'movie-shot-trailers', 'trailer', 'tt0444850', 'shot_0016.mp4']
            if 'shot_' in file.key:
                path = os.path.join(file.key.split('/')[1], file.key.split('/')[2], file.key.split('/')[3])
                full_path = os.path.join(path, file.key.split('/')[4])
                if not os.path.exists(path):
                    os.makedirs(path)
                if not os.path.exists(full_path):
                    with open(full_path, 'wb') as data:
                        bucket.download_fileobj(file.key, data)
            else:
                if not os.path.exists('movie-shot-trailers/extras'):
                    os.makedirs('movie-shot-trailers/extras')
                with open(f'movie-shot-trailers/extras/{os.path.split(file.key)[1]} ', 'wb') as data:
                    bucket.download_fileobj(file.key, data)
    else:
        i = 0
        for file in bucket.objects.filter(Prefix = 'shot-type/movie-shot-trailers'):
            i += 1
            if i == quantity:
                break
            if 'shot_' in file.key:
                path = os.path.join(file.key.split('/')[1], file.key.split('/')[2], file.key.split('/')[3])
                full_path = os.path.join(path, file.key.split('/')[4])
                if not os.path.exists(path):
                    os.makedirs(path)
                if not os.path.exists(full_path):
                    with open(full_path, 'wb') as data:
                        bucket.download_fileobj(file.key, data)
            else:
                if not os.path.exists('movie-shot-trailers/extras'):
                    os.makedirs('movie-shot-trailers/extras')
                with open(f'movie-shot-trailers/extras/{os.path.split(file.key)[1]} ', 'wb') as data:
                    bucket.download_fileobj(file.key, data)


def saliency(image, method=None, show=False):
    # load the input image
    # image = cv2.imread(image)
    if method == 'residual':
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        print(success)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        if show is not False:
            cv2.imshow("Image", image)
            cv2.imshow("Output", saliencyMap)
            cv2.waitKey(0)
        return saliencyMap
    if method == 'finegrained':
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        if show is not False:
            # show the images
            cv2.imshow("Image", image)
            cv2.imshow("Output", saliencyMap)
            cv2.waitKey(0)
        return saliencyMap
    if method == 'fgthreshold':
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        saliencyMap = (saliencyMap * 255).astype("uint8")
        threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if show is not False:
            # show the images
            cv2.imshow("Image", image)
            cv2.imshow("Thresh", threshMap)
            cv2.waitKey(0)
        return threshMap
    if method is None:
        pass
    else:
        raise ValueError('Selected method is incorrect')

# download_data(quantity=5)

# saliency('Team3_FP/utils/ok.png', method='finegrain', show='True')

