import numpy as np
from tiramisu import *
import keras
import sys
import argparse
import os
from PIL import Image
from scipy.misc import imresize
from preprocessing import load_image
from params import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument

add_arg('images', nargs='*', default=[])
add_arg('--model', default=MODEL_PATH + 'wally_5000_6_67.h5', type=str)
add_arg('--output_path', default='output', type=str)
add_arg('--img_size', default=(2800, 1760), type=tuple, help='resolution to load images')
args = parser.parse_args()


def img_resize(img):
    h, w, _ = img.shape
    nvpanels = h / 160
    nhpanels = w / 160
    new_h, new_w = h, w
    if nvpanels * 160 != h:
        new_h = (nvpanels + 1) * 160
    if nhpanels * 160 != w:
        new_w = (nhpanels + 1) * 160
    if new_h == h and new_w == w:
        return img
    else:
        return (imresize(img, (new_h, new_w)) / 255. - mu) / std


def split_panels(img):
    h, w, _ = img.shape
    num_vert_panels =   int(h / 160)
    num_hor_panels =    int(w / 160)
    panels = []
    for i in range(num_vert_panels):
        for j in range(num_hor_panels):
            panels.append(img[i * 160:(i + 1) * 160, j * 160:(j + 1) * 160])
    return np.stack(panels)


def combine_panels(img, panels):
    h, w, _ = img.shape
    num_vert_panels = int(h / 160)
    num_hor_panels =  int(w / 160)
    total = []
    p = 0
    for i in range(num_vert_panels):
        row = []
        for j in range(num_hor_panels):
            row.append(panels[p])
            p += 1
        total.append(np.concatenate(row, axis=1))
    return np.concatenate(total, axis=0)


def prediction_mask(img, target):
    layer1 = Image.fromarray(((img * std + mu) * 255).astype('uint8'))
    layer2 = Image.fromarray(
        np.concatenate(
            4 * [np.expand_dims((161 * (1 - target)).astype('uint8'), axis=-1)],
            axis=-1))
    result = Image.new("RGBA", layer1.size)
    result = Image.alpha_composite(result, layer1.convert('RGBA'))
    return Image.alpha_composite(result, layer2)


def waldo_predict(img):
    rimg = img_resize(img)
    panels = split_panels(rimg)
    pred_panels = model.predict(panels, batch_size=6)
    pred_panels = np.stack([reshape_pred(pred) for pred in pred_panels])
    return rimg, combine_panels(rimg, pred_panels)


def reshape_pred(pred): return pred.reshape(160, 160, 2)[:, :, 1]


if __name__ == "__main__":
    """
    This script makes predictions on a list of inputs with a pre-trained model,
    and saves them as transparency masks over the original image.
    # Example:
    $ python predict.py image1.jpg image2.jpg
    """
    images = args.images
    img_sz = args.img_size
    print(args.images)

    input_shape = (160, 160, 3)

    img_input = Input(shape=input_shape)
    x = create_tiramisu(2, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
    model = Model(img_input, x)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(1e-3),
                  metrics=["accuracy"],
                  sample_weight_mode='temporal')

    model.load_weights(args.model)
    for i, image in enumerate(images):
        full_image = load_image(image, img_sz)
        full_image_r, full_pred = waldo_predict(full_image)
        mask = prediction_mask(full_image_r, full_pred)
        mask.save(os.path.join(args.output_path, 'output_' + str(i) + '.png'))
