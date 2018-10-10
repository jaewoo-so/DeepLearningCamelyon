#! /usr/bin/env python3
import cv2
from scipy.misc import imsave
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
import glob
#before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

slide_path = '/home/wli/Downloads/CAMELYON16/training/tumor'

Heatmap_path = '/home/wli/Downloads/pred/realheatmap_bbox/'
#BASE_TRUTH_DIR = '/raida/wjc/CAMELYON16/training/masking'
BASE_TRUTH_DIR = '/home/wli/Downloads/CAMELYON16/masking'
dimension_path = '/home/wli/Downloads/pred/dimensions/'
slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
slide_paths.sort()
BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
BASE_TRUTH_DIRS.sort()
Heat_map_paths = glob.glob(osp.join(Heatmap_path, '*.npy'))
Heat_map_paths.sort()
dimension_paths = glob.glob(osp.join(dimension_path, '*.npy'))
dimension_paths.sort()

#image_pair = zip(tumor_paths, anno_tumor_paths)  
#image_pair = list(image_mask_pair)
sampletotal= pd.read_pickle('/home/wli/Downloads/false_positive_patches/false_positive_patch_index.pkl')

false_positive_patch_path = '/home/wli/Downloads/false_positive_patches'


NUM_CLASSES = 2 # not_tumor, tumor

#sampletotal = np.load('/home/wli/Downloads/false_positive_patches/falsepositivepatches.npy')

def gen_imgs_false_positive(samples, batch_size, base_truth_dir=BASE_TRUTH_DIR, shuffle=False):
   
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1) # shuffle samples
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
        
            #images = []
            #masks = []
            for _, batch_sample in batch_samples.iterrows():
                slide_contains_tumor = osp.basename(batch_sample.slide_path).startswith('tumor_')
                 
                with openslide.open_slide(batch_sample.slide_path) as slide:
                    tiles = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)
                    img = tiles.get_tile(tiles.level_count-1, batch_sample.tile_loc[::-1])
                    im = np.array(img)
                    int1, int2= batch_sample.tile_loc[::-1]
                    imsave('/home/wli/Downloads/false_positive_patches/%s_%d_%d.png' % (os.path.splitext(osp.basename(batch_sample.slide_path))[0], int1, int2), im)

                # only load truth mask for tumor slides
                if slide_contains_tumor:
                    truth_slide_path = osp.join(base_truth_dir, osp.basename(batch_sample.slide_path).replace('.tif', '_mask.tif'))
                    with openslide.open_slide(str(truth_slide_path)) as truth:
                        truth_tiles = DeepZoomGenerator(truth, tile_size=224, overlap=0, limit_bounds=False)
                        mask = truth_tiles.get_tile(truth_tiles.level_count-1, batch_sample.tile_loc[::-1])
                        mask = (cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY) > 0).astype(int)
                        mk = np.array(mask)
                        int1, int2= batch_sample.tile_loc[::-1]
                        imsave('/home/wli/Downloads/false_positive_patches/mask_%s_%d_%d.png' % (os.path.splitext(osp.basename(batch_sample.slide_path))[0], int1, int2), mk)
                else:
                    mask = np.zeros((224, 224))
                    mk = np.array(mask)
                    int1, int2= batch_sample.tile_loc[::-1]
                    imsave('/home/wli/Downloads/false_positive_patches/mask_%s_%d_%d.png' % (os.path.splitext(osp.basename(batch_sample.slide_path))[0], int1, int2), mk)

            yield


next(gen_imgs_false_positive(sampletotal, len(sampletotal)))
