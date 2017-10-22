import datetime

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


def run():
    # Load Data
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
    sample_weights[:, :, 0] = weights[0]
    sample_weights[:, :, 1] = weights[1]
    input_shape = (160, 160, 3)

    # Create Model
    img_input = Input(shape=input_shape)
    x = create_tiramisu(2, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
    model = Model(img_input, x)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"],
                  sample_weight_mode='temporal')

    # Load Model Weights
    if (params.LOAD_MODEL !=""):
        model.load_weights(params.LOAD_MODEL)

    # Training
    if(params.SPLIT == 0):
        gen_mix = seg_gen_mix(waldo_sub_imgs, waldo_sub_labels, images, labels)
    else:
        gen_mix = seg_gen_mix(waldo_sub_imgs, waldo_sub_labels, images, labels, tot_bs=6, prop=params.SPLIT)
    a = datetime.datetime.now()
    model.fit_generator(gen_mix, steps_per_epoch=params.STEPS_PER_EPOCH, epochs=params.EPOCHS, verbose=0, callbacks=[TQDMCallback()],
                        class_weight=sample_weights)
    b = datetime.datetime.now()
    print('Training Complete! Time: ', b - a)

    # Save model
    model.save(params.SAVE_MODEL + ".h5")

    # View Results
    h = model.history.history
    plt.plot(h['loss'])
    plt.show()


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
    sample_weights[:, :, 0] = weights[0]
    sample_weights[:, :, 1] = weights[1]
    input_shape = (160, 160, 3)

    img_input = Input(shape=input_shape)
    x = create_tiramisu(2, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
    model = Model(img_input, x)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"],
                  sample_weight_mode='temporal')
    spe = 6
    ep = 10
    a = datetime.datetime.now()
    model.fit_generator(gen_mix, steps_per_epoch=spe, epochs=ep, verbose=0, callbacks=[TQDMCallback()],
                        class_weight=sample_weights)
    b = datetime.datetime.now()
    print('Training Complete! Time: ', b - a)

    # Save model
    modelName = 'model_' + str(ep) + 'epochs.h5'
    model.save(modelName)

