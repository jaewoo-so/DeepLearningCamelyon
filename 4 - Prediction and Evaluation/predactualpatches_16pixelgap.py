from matplotlib import cm
from tqdm import tqdm
from skimage.filters import threshold_otsu
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
import glob
import math
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
from keras.utils.np_utils import to_categorical

output_dir = Path('/home/wli/Downloads/camelyontestonly')

import os.path as osp
import openslide
from pathlib import Path
from keras.models import Sequential
from keras.layers import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

#BASE_TRUTH_DIR = Path('/home/wli/Downloads/camelyontest/mask')

#slide_path = '/home/wli/Downloads/CAMELYON16/training/tumor/'
slide_path = '/home/wli/Downloads/googlepred/'

#slide_path = '/home/wli/Downloads/CAMELYON16/training/normal/'

#slide_path_validation = '/home/wli/Downloads/CAMELYON16/training/tumor/validation/'
#slide_path_validation = '/home/wli/Downloads/CAMELYON16/training/normal/validation/'
#truth_path = str(BASE_TRUTH_DIR / 'tumor_026_Mask.tif')
#slide_paths = list(slide_path)

slide_paths = glob.glob(osp.join(slide_path, '*.tif'))

#index_path = '/Users/liw17/Documents/pred_dim/normal/'
index_path = '/home/wli/Downloads/predpatches/'
index_paths = glob.glob(osp.join(index_path, '*.pkl'))


#slide_paths_validation = glob.glob(osp.join(slide_path_validation, '*.tif'))

#slide_paths = slide_paths + slide_paths_validation
#slide_paths = slide_path

# slide_paths.sort()

#slide = openslide.open_slide(slide_path)


NUM_CLASSES = 2  # not_tumor, tumor


def gen_imgs(samples, batch_size, slide, shuffle=False):
    """This function returns a generator that 
    yields tuples of (
        X: tensor, float - [batch_size, 224, 224, 3]
        y: tensor, int32 - [batch_size, 224, 224, NUM_CLASSES]
    )
    input: samples: samples dataframe
    input: batch_size: The number of images to return for each pull
    output: yield (X_train, y_train): generator of X, y tensors
    option: base_truth_dir: path, directory of truth slides
    option: shuffle: bool, if True shuffle samples
    """

    num_samples = len(samples)
    print(num_samples)

    images = []

    for _, batch_sample in batch_samples.iterrows():

        #tiles = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)
        #xy = batch_sample.tile_loc[::]
        xy = batch_sample.tile_loc[::-1]
        xylarge = [x * 224 for x in xy]
        print(batch_sample.tile_loc[::-1], xylarge)
        #img = tiles.get_tile(tiles.level_count-1, batch_sample.tile_loc[::-1])
        for m in range(0, 224, 16):
            for n in range(0, 224, 16):
                img = slide.read_region((xylarge[0]+112+n, xylarge[1]+112+m), 0, crop_size)
                img = np.array(img)
                img = img[:, :, :3]

                images.append(img)

    X_train = np.array(images)

    yield X_train


def predict_batch_from_model(patches, model):

    predictions = model.predict(patches)
    #print(predictions[:, 1])
    #print(predictions[:, 0])
    predictions = predictions[:, 1]
    return predictions


model = load_model(
    '/home/wli/Downloads/googlenet0917-02-0.93.hdf5')
alpha = 0.5

#slide = openslide.open_slide(slide_paths[0])

#n_cols = int(slide.dimensions[0] / 224)
#n_rows = int(slide.dimensions[1] / 224)
#assert n_cols * n_rows == n_samples

#thumbnail = slide.get_thumbnail((n_cols, n_rows))
#thumbnail = np.array(thumbnail)

# batch_size = n_cols
batch_size = 32


crop_size = [224, 224]
i = 0
while i < len(slide_paths):

    output_thumbnail_preds = list()
 #   all_samples = find_patches_from_slide(
 #      slide_paths[i], filter_non_tissue=False)
    all_samples = pd.read_pickle(index_paths[i])
    all_samples.slide_path = slide_paths[i]
    print(all_samples)
    n_samples = len(all_samples)
    slide = openslide.open_slide(slide_paths[i])
    
    for offset in tqdm(list(range(0, n_samples, batch_size))):
        batch_samples = all_samples.iloc[offset:offset+batch_size]
        #png_fnames = batch_samples.tile_loc.apply(lambda coord: str(output_dir / ('%d_%d.png' % coord[::-1])))

        X = next(gen_imgs(batch_samples, batch_size, slide, shuffle=False))

        preds = predict_batch_from_model(X, model)
        

        output_thumbnail_preds.append(preds)
        #output_thumbnail_preds.extend(preds)
        # print(output_thumbnail_preds)

        # overlay preds
        # save blended imgs
        # for i, png_fname in enumerate(png_fnames):
        #    pred_i = preds[i]
        #    X_i = X[i]
        #    output_img = cv2.cvtColor(X_i, cv2.COLOR_RGB2GRAY)
        #    output_img2 = cv2.cvtColor(output_img.copy(), cv2.COLOR_GRAY2RGB)

        #    overlay = np.uint8(cm.jet(pred_i) * 255)[:,:,:3]
        #    blended = cv2.addWeighted(overlay, alpha, output_img2, 1-alpha, 0, output_img)

        #plt.imsave(png_fname, blended)

    #output_thumbnail_preds = np.array(output_thumbnail_preds)

    np.save('/home/wli/Downloads/googlepred/%s' %
            (osp.splitext(osp.basename(slide_paths[i]))[0]), output_thumbnail_preds)

    i = i+1
