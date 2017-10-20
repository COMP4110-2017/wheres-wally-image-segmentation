import glob
import datetime

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
import tensorflow as tf
from tiramisu import *
import keras
from keras_tqdm import TQDMCallback

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if __name__ == "__main__":
    images = np.load('images.npy')
    labels = np.load('labels.npy')
    waldo_sub_imgs = np.load('wally_sub_images.npy')
    waldo_sub_labels = np.load('wally_sub_labels.npy')
    gen_mix = seg_gen_mix(waldo_sub_imgs, waldo_sub_labels, images, labels)
    X, y = next(gen_mix)
    freq0 = float(np.sum(labels == 0))
    freq1 = float(np.sum(labels == 1))
    weights = {0: 1 / freq0, 1: 1.}
    sample_weights = np.zeros((X.shape[0], X.shape[1] * X.shape[2], 2))
    sample_weights.shape
    sample_weights[:, :, 0] = weights[0]
    sample_weights[:, :, 1] = weights[1]
    input_shape = (160, 160, 3)

    img_input = Input(shape=input_shape)
    x = create_tiramisu(2, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
    model = Model(img_input, x)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"],
                  sample_weight_mode='temporal')
    a = datetime.datetime.now()
    ep = 1
    model.fit_generator(gen_mix, steps_per_epoch=5, epochs=ep, verbose=0, callbacks=[TQDMCallback()],
                        class_weight=sample_weights)
    b = datetime.datetime.now()
    print('Training Complete! Time: ', b - a)

    # Save model
    modelName = 'model_' + str(ep) + 'epochs.h5'
    model.save(modelName)

    # Display error history

    # h = model.history.history
    # plt.plot(h['loss'])
    # plt.show()

    # Predict on new image
    gen_mix = seg_gen_mix(waldo_sub_imgs, waldo_sub_labels, images, labels)
    X, y = next(gen_mix)
    # plt.imshow(X[1] * std + mu)
    # plt.show()
    # pred = model.predict_on_batch(X)
    # plt.imshow(pred[1].reshape(160, 160, 2)[:, :, 1])
    # plt.show()

    gen_mix = seg_gen_mix(waldo_sub_imgs, waldo_sub_labels, images, labels, tot_bs=6, prop=.67)

    ep = 1
    model.fit_generator(gen_mix, steps_per_epoch=3, epochs=ep, verbose=0, callbacks=[TQDMCallback()],
                        class_weight=sample_weights)

    # Save model
    modelName = 'model_' + str(ep) + 'epochs.h5'
    model.save(modelName)

    # Display error history

    # h = model.history.history
    # plt.plot(h['loss'])
    # plt.show()

    model.fit_generator(gen_mix, steps_per_epoch=6, epochs=1, verbose=0, callbacks=[TQDMCallback()],
                        class_weight=sample_weights)

    ep = 5000
    model.fit_generator(gen_mix, steps_per_epoch=6, epochs=ep, verbose=0, callbacks=[TQDMCallback()],
                        class_weight=sample_weights)

    modelName = 'model_' + str(ep) + 'epochs.h5'
    model.save(modelName)

    h = model.history.history
    plt.plot(h['loss'])
    plt.show()

    plt.imshow(X[1] * std + mu)
    plt.show()
    pred = model.predict_on_batch(X)
    plt.imshow(pred[1].reshape(160, 160, 2)[:, :, 1])
    plt.show()
    model.save('overnight.h5')
    model.save('overnight_backup1.h5')
    model.save('overnight_backup2.h5')
    model.save('overnight_backup3.h5')