#! /usr/bin/env python3
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
from pandas import HDFStore
#import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

print('Hi, patch extraction can take a while, please be patient...')
#slide_path = '/raida/wjc/CAMELYON16/training/tumor'
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

false_positive_patch_path = '/home/wli/Downloads/false_positive_patches'

sampletotal = pd.DataFrame([])
i=0
while i < len(slide_paths):
    #sampletotal = pd.DataFrame([])
    base_truth_dir = Path(BASE_TRUTH_DIR)
    slide_contains_tumor = osp.basename(slide_paths[i]).startswith('tumor_')

    #pred = np.load('realheatmap.npy')
    pred = np.load(Heat_map_paths[i])

    pred_dim = np.load(dimension_paths[i])

    pred_binary = (pred > 0.5)*1

    pred_patches = pd.DataFrame(pd.DataFrame(pred_binary).stack())

########################################################################################
## make a new column to record if a patch is a tumor
########################################################################################
    pred_patches['pred_is_tumor'] = pred_patches[0]

    pred_patches.reset_index(inplace=True, drop=True)



    with openslide.open_slide(slide_paths[i]) as slide:

        thumbnail = slide.get_thumbnail((slide.dimensions[0] / 224, slide.dimensions[1] / 224))

        #patches = pd.DataFrame(pd.DataFrame(np.array(thumbnail.convert("L"))).stack())
        patches = pd.DataFrame(np.array(thumbnail.convert("L")))
        
        samplesforpred = patches.loc[pred_dim[5]:pred_dim[6], pred_dim[3]:pred_dim[4]]

        #patches = pd.DataFrame(pd.DataFrame(binary).stack())

        samplesforpredfinal = pd.DataFrame(samplesforpred.stack())


        samplesforpredfinal['tile_loc'] = list(samplesforpredfinal.index)



        samplesforpredfinal.reset_index(inplace=True, drop=True)


        samplesforpredfinal['slide_path'] = slide_paths[i]

        
    
    if slide_contains_tumor:

        truth_slide_path = base_truth_dir / osp.basename(slide_paths[i]).replace('.tif', '_mask.tif')

        with openslide.open_slide(str(truth_slide_path)) as truth:
            thumbnail_truth = truth.get_thumbnail((truth.dimensions[0] / 224, truth.dimensions[1] / 224))

            #thumbnail_truth_gray = np.array(thumbnail_truth.convert("L"))
            thumbnail_truth_gray = pd.DataFrame(np.array(thumbnail_truth.convert("L")))

            #thumbnail_truth_roi = thumbnail_truth_gray[pred_dim[5]:pred_dim[6], pred_dim[3]:pred_dim[4]]
            thumbnail_truth_roi = thumbnail_truth_gray.loc[pred_dim[5]:pred_dim[6], pred_dim[3]:pred_dim[4]]

        #patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth_roi.convert("L"))).stack())
        patches_y = pd.DataFrame(thumbnail_truth_roi.stack())

        patches_y['is_tumor'] = (patches_y[0] > 0)*1

        patches_y.reset_index(inplace=True, drop=True)


        #patches['slide_path'] = slide_paths[i]

        
        #samples = pd.concat([patches, patches_y, pred_patches], axis=1)

        #samples['tile_loc'] = list(samples.index)
        #samples.reset_index(inplace=True, drop=True)

        #samples = samples[samples.pred_is_tumor > samples.is_tumor]
        samplesforpredfinal['is_tumor'] = patches_y['is_tumor']
        samplesforpredfinal['pred_is_tumor'] = pred_patches['pred_is_tumor']
        samples = samplesforpredfinal[samplesforpredfinal.pred_is_tumor > samplesforpredfinal.is_tumor]
        
    else:

        #samples = pred_patches

        samplesforpredfinal['pred_is_tumor'] = pred_patches['pred_is_tumor']

        #samples['tile_loc'] = list(samples.index)

        #samples.reset_index(inplace=True, drop=True)
        samples = samplesforpredfinal[samplesforpredfinal.pred_is_tumor > 0]

        #patches['is_tumor'] = False
        #sampletotal.append(patches)
            
       
        
        
    sampletotal=sampletotal.append(samples, ignore_index=True)
        
    i=i+1
        

#np.save('/home/wli/Downloads/false_positive_patches/falsepositivepatches', sampletotal)


#store = HDFStore('/home/wli/Downloads/false_positive_patches/false_positive_patch_index.h5')

#store['sampletotal'] = sampletotal  # save it

sampletotal.to_pickle('/home/wli/Downloads/false_positive_patches/false_positive_patch_index.pkl')

