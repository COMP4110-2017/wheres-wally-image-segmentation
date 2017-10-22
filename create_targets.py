# Imports

import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from keras.preprocessing import image
import os
import glob
from numpy import random
import scipy
from PIL import Image
from params import *

# Helper Functions

# get image number
def get_image_number(image_file):
    return os.path.basename(image_file).split('.')[0]


# get xml file path corresponding to image file
def grab_xml_file(image_file):
    i = get_image_number(image_file)
    return XML_PATH + i + '.xml'


# extract bounding boxes from xml
def decode_bounding_box(xml):
    soup = BeautifulSoup(xml, 'xml')
    boxes = []
    for box in soup.annotation.find_all('bndbox'):
        boxes.append((int(box.xmin.contents[0]), int(box.xmax.contents[0]),
                      int(box.ymin.contents[0]), int(box.ymax.contents[0])))
        return boxes

# creates target image arrays and saves as .png files
def make_target(img_file, boxes):
    img = Image.open(img_file)
    img_array = image.img_to_array(img, data_format='channels_last')
    shape = img_array.shape
    target = np.zeros(shape)
    for box in boxes:
        xmin, xmax, ymin, ymax = box
        target[ymin:ymax,xmin:xmax, :] = 1
    return target

# iterate through the files to create and save target images
def run():
    img_files = glob.glob(IMAGE_PATH + "*.jpg")
    if not os.path.exists(TARGET_PATH):
        os.mkdir(TARGET_PATH)
    for img in img_files:
        xml_file = grab_xml_file(img)
        i = get_image_number(xml_file)
        with open(xml_file, mode='r') as f:
            raw_xml = f.read()
        boxes = decode_bounding_box(raw_xml)
        target = make_target(img, boxes)
        scipy.misc.imsave(TARGET_PATH + i + '.png', target)

if __name__ == "__main__":
    img_files = glob.glob(IMAGE_PATH+"*.jpg")
    if not os.path.exists(TARGET_PATH):
        os.mkdir(TARGET_PATH)
    for img in img_files:
        xml_file = grab_xml_file(img)
        i = get_image_number(xml_file)
        with open(xml_file, mode='r') as f:
            raw_xml = f.read()
        boxes = decode_bounding_box(raw_xml)
        target = make_target(img, boxes)
        scipy.misc.imsave(TARGET_PATH+i+'.png', target)