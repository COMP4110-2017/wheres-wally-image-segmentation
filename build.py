import glob
from PIL import Image
import os
from keras.preprocessing import image
# from keras.utils import to_categorical
import threading
from numpy import random
from bs4 import BeautifulSoup
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os

from params import *
from create_targets import *
from preprocessing import *
from generate_subimages import *
from generators import *

if __name__=="__main__":

    images = np.load('images.npy')
    labels = np.load('labels.npy')

    waldo_sub_imgs = np.load('waldo_sub_imgs.npy')
    waldo_sub_labels = np.load('waldo_sub_labels.npy')

    gen_mix = seg_gen_mix(waldo_sub_imgs, waldo_sub_labels, images, labels)

    X, y = next(gen_mix)

    print(X.shape)

    '''
    freq0 = float(np.sum(labels==0))
    
    freq1 = float(np.sum(labels==1))
    '''