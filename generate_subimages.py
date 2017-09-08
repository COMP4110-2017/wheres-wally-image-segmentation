import numpy as np
from params import *
import matplotlib.pyplot as plt


def extract_sub_image(img, box, side_length):
    xStart, xEnd, yStart, yEnd = box
    i = img[xEnd-224:xStart+side_length, yEnd-side_length:yStart+side_length]
    if i.shape[0]==0:
        return img[xStart:xStart+(side_length+1), yEnd-side_length:yStart+side_length]
    else:
        return i


def find_box(label):
    """
    Finds the bounding box given input label.
    """
    shp = label.shape
    yStart = np.argmax(label.sum(axis=0))
    yEnd = int(yStart+np.max(np.unique(label.sum(axis=1))))
    xStart = np.argmax(label.sum(axis=1))
    xEnd= int(xStart+np.max(np.unique(label.sum(axis=0))))
    return xStart, xEnd, yStart, yEnd


if __name__=="__main__":
    side_length = 224
    imgs = np.load('imgs.npy')
    labels = np.load('labels.npy')
    waldo_sub_imgs = []
    waldo_sub_labels = []
    for i, label in enumerate(labels):
        box = find_box(label)
        waldo_sub_imgs.append(extract_sub_image(imgs[i], box, side_length))
        waldo_sub_labels.append(extract_sub_image(label, box, side_length))
    np.save('wally_sub_imgs.npy',np.array(waldo_sub_imgs))
    np.save('wally_sub_labels.npy',np.array(waldo_sub_labels))