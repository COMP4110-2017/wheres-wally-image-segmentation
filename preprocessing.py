# imports
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from params import *

def load_image(image_file,image_size=None):
    if image_size:
        return np.array(Image.open(image_file).resize(image_size,Image.NEAREST))
    else:
        return np.array(Image.open(image_file))

def load_label(image_file, image_size):
    return np.array(Image.open(image_file).convert("L").resize(image_size,Image.NEAREST))

def run():
    # load the origional image files
    get_image_number = lambda image_file: int(os.path.basename(image_file).split('.')[0])
    target_files = sorted(glob.glob(TARGET_PATH + "*.png"), key=get_image_number)
    image_files = sorted(glob.glob(IMAGE_PATH + "*.jpg"), key=get_image_number)

    # set image size to largest image in set
    image_size = (2800, 1760)

    images = np.stack([load_image(image_file, image_size) for image_file in image_files])
    labels = np.stack([load_label(target_file, image_size) for target_file in target_files])

    # need to normalize pixels to values from 0-255
    images = images / 255
    labels = labels / 255

    mu = np.mean(images)

    std = np.std(images)

    images -= mu
    images /= std

    np.save('images.npy', images)
    np.save('labels.npy', labels)

    images = np.load('images.npy')
    labels = np.load('labels.npy')


if __name__=="__main__":
    # load the origional image files
    get_image_number = lambda image_file: int(os.path.basename(image_file).split('.')[0])
    target_files = sorted(glob.glob(TARGET_PATH+"*.png"), key=get_image_number)
    image_files = sorted(glob.glob(IMAGE_PATH+"*.jpg"), key=get_image_number)

    # set image size to largest image in set
    image_size = (2800, 1760)

    images = np.stack([load_image(image_file,image_size) for image_file in image_files])
    labels = np.stack([load_label(target_file,image_size) for target_file in target_files])

    # need to normalize pixels to values from 0-255
    images = images/255
    labels = labels/255

    mu = np.mean(images)

    std = np.std(images)

    images -= mu
    images /= std

    np.save('images.npy', images)
    np.save('labels.npy', labels)

    images = np.load('images.npy')
    labels = np.load('labels.npy')

    '''
    plt.imshow(images[-2] * std + mu)
    plt.show()
    plt.imshow(labels[-2] * std + mu)
    plt.show()
    '''
