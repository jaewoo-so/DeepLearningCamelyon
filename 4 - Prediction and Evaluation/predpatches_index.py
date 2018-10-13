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
slide_path = '/Volumes/WZL-NIAID-5/New folder (4)/CAMELYON16/training/normal/'

#slide_path = '/home/wli/Downloads/CAMELYON16/training/normal/'

#slide_path_validation = '/home/wli/Downloads/CAMELYON16/training/tumor/validation/'
#slide_path_validation = '/home/wli/Downloads/CAMELYON16/training/normal/validation/'
#truth_path = str(BASE_TRUTH_DIR / 'tumor_026_Mask.tif')
#slide_paths = list(slide_path)

slide_paths = glob.glob(osp.join(slide_path, '*.tif'))

#slide_paths_validation = glob.glob(osp.join(slide_path_validation, '*.tif'))

#slide_paths = slide_paths + slide_paths_validation
#slide_paths = slide_path

# slide_paths.sort()

#slide = openslide.open_slide(slide_path)


def find_patches_from_slide(slide_path, filter_non_tissue=True):
    """Returns a dataframe of all patches in slide
    input: slide_path: path to WSI file
    output: samples: dataframe with the following columns:
        slide_path: path of slide
        is_tissue: sample contains tissue
        is_tumor: truth status of sample
        tile_loc: coordinates of samples in slide


    option: base_truth_dir: directory of truth slides
    option: filter_non_tissue: Remove samples no tissue detected
    """

    #sampletotal = pd.DataFrame([])
    #base_truth_dir = Path(BASE_TRUTH_DIR)
    #anno_path = Path(anno_path)
    #slide_contains_tumor = osp.basename(slide_paths[i]).startswith('tumor_')
    print(slide_path)

    dimensions = []

    with openslide.open_slide(slide_path) as slide:
        dtotal = (slide.dimensions[0] / 224, slide.dimensions[1] / 224)
        thumbnail = slide.get_thumbnail((dtotal[0], dtotal[1]))
        thum = np.array(thumbnail)
        ddtotal = thum.shape
        dimensions.extend(ddtotal)
        hsv_image = cv2.cvtColor(thum, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
        # be min value for v can be changed later
        minhsv = np.array([hthresh, sthresh, 70], np.uint8)
        maxhsv = np.array([180, 255, vthresh], np.uint8)
        thresh = [minhsv, maxhsv]
        print(thresh)
        # extraction the countor for tissue

        rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
        _, contours, _ = cv2.findContours(
            rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
        bboxt = pd.DataFrame(columns=bboxtcols)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            bboxt = bboxt.append(
                pd.Series([x, x+w, y, y+h], index=bboxtcols), ignore_index=True)
            bboxt = pd.DataFrame(bboxt)

        xxmin = list(bboxt['xmin'].get_values())
        xxmax = list(bboxt['xmax'].get_values())
        yymin = list(bboxt['ymin'].get_values())
        yymax = list(bboxt['ymax'].get_values())

        xxxmin = np.min(xxmin)  # xxxmin = math.floor((np.min(xxmin))*224)
        xxxmax = np.max(xxmax)  # xxxmax = math.floor((np.max(xxmax))*224)
        yyymin = np.min(yymin)  # yyymin = math.floor((np.min(yymin))*224)
        yyymax = np.max(yymax)  # yyymax = math.floor((np.max(yymax))*224)

        dcoord = (xxxmin, xxxmax, yyymin, yyymax)
        print(dcoord)
        dimensions.extend(dcoord)

       # bboxt = math.floor(np.min(xxmin)*256), math.floor(np.max(xxmax)*256), math.floor(np.min(yymin)*256), math.floor(np.max(yymax)*256)

        samplesnew = pd.DataFrame(pd.DataFrame(
            np.array(thumbnail.convert('L'))))
        print(samplesnew)
        # very critical: y value is for row, x is for column
        samplesforpred = samplesnew.loc[yyymin:yyymax, xxxmin:xxxmax]
        #samplesforpred2 = samplesforpred*224
        dsample = samplesforpred.shape

        dimensions.extend(dsample)
        print(dimensions)
        np.save('/Users/liw17/Documents/pred_dim/normal/dimensions_%s' %
                (osp.splitext(osp.basename(slide_paths[i]))[0]), dimensions)

        # print(samplesforpred)

        samplesforpredfinal = pd.DataFrame(samplesforpred.stack())

        print(samplesforpredfinal)

        samplesforpredfinal['tile_loc'] = list(samplesforpredfinal.index)

        samplesforpredfinal.reset_index(inplace=True, drop=True)

        samplesforpredfinal['slide_path'] = slide_paths[i]

        print(samplesforpredfinal)

        samplesforpredfinal.to_pickle(
            '/Users/liw17/Documents/pred_dim/normal/patch_index_%s.pkl' % (osp.splitext(osp.basename(slide_path))[0]))

    return samplesforpredfinal


i = 0
while i < len(slide_paths):

    find_patches_from_slide(
        slide_paths[i], filter_non_tissue=False)

    i = i+1
