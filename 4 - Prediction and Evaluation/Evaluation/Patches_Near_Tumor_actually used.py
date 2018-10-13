import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import openslide
from pathlib import Path
from scipy.misc import imsave as saveim
from skimage.filters import threshold_otsu
import glob
# before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
import cv2 as cv2
from skimage import io
import xml.etree.ElementTree as et
import pandas as pd
import math
import random

# setup for path
#BASE_TRUTH_DIR = Path('/home/ubuntu/data/Ground_Truth_Extracted/Mask')

#slide_path = '/home/ubuntu/data/slides/Tumor_009.tif'
#truth_path = str(BASE_TRUTH_DIR / 'Tumor_009_Mask.tif')
#BASE_TRUTH_DIR = Path('/Users/liw17/Downloads/camelyontest/Mask')

#slide_path = '/Users/liw17/Downloads/camelyontest/slides/tumor_026.tif'
#truth_path = osp.join(BASE_TRUTH_DIR, 'tumor_026_mask.tif')


#slide = openslide.open_slide(slide_path)
#truth = openslide.open_slide(truth_path)
print('Hi, patch extraction can take a while, please be patient...')
slide_path = '/home/wli/Downloads/CAMELYON16/training/tumor'
slide_path_normal = '/home/wli/Downloads/CAMELYON16/training/normal'

slide_path = '/Users/liw17/Documents/camelyon16/train/tumor/'
truth_path = '/Users/liw17/Documents/camelyon16/train/mask/'

anno_path = '/Users/liw17/Documents/camelyon16/train/'
#anno_path = '/home/wli/Downloads/CAMELYON16/training/Lesion_annotations'
BASE_TRUTH_DIR = '/Users/liw17/Documents/camelyon16/train/mask/'
slide_paths = glob.glob(osp.join(slide_path, '*.tif'))
slide_paths.sort()
slide_paths_normal = glob.glob(osp.join(slide_path_normal, '*.tif'))
slide_paths_normal.sort()
slide_paths_total = slide_paths
#slide_paths_total = slide_paths + slide_paths_normal
BASE_TRUTH_DIRS = glob.glob(osp.join(BASE_TRUTH_DIR, '*.tif'))
Anno_paths = glob.glob(osp.join(anno_path, '*.xml'))
BASE_TRUTH_DIRS.sort()

#image_pair = zip(tumor_paths, anno_tumor_paths)
#image_pair = list(image_mask_pair)

# for color normalization
#lut = np.asarray(Image.open(filename)).squeeze()
# def apply_lut(tile, lut):
#  """ Apply look-up-table to tile to normalize H&E staining. """
#  ps = tile.shape # tile size is (rows, cols, channels)
#  reshaped_tile = tile.reshape((ps[0]*ps[1], 3))
#  normalized_tile = np.zeros((ps[0]*ps[1], 3))
#  idxs = range(ps[0]*ps[1])
#  Index = 256 * 256 * reshaped_tile[idxs,0] + 256 * reshaped_tile[idxs,1] + reshaped_tile[idxs,2]
#  normalized_tile[idxs] = lut[Index.astype(int)]
#  return normalized_tile.reshape(ps[0], ps[1], 3).astype(np.uint8)


# go through all the file


def convert_xml_df(file):
    parseXML = et.parse(file)
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    df_xml = pd.DataFrame(columns=dfcols)
    for child in root.iter('Annotation'):
        for coordinate in child.iter('Coordinate'):
            Name = child.attrib.get('Name')
            Order = coordinate.attrib.get('Order')
            X_coord = float(coordinate.attrib.get('X'))
            Y_coord = float(coordinate.attrib.get('Y'))
            df_xml = df_xml.append(
                pd.Series([Name, Order, X_coord, Y_coord], index=dfcols), ignore_index=True)
            df_xml = pd.DataFrame(df_xml)
    return (df_xml)


def random_crop(slide, truth, crop_size, bbox):

    #width, height = slide.level_dimensions[0]
    dy, dx = crop_size
    x, y = bbox
    x = int(x)
    y = int(y)
    print(x, y)
    index = [[x, y], [x, y+dy-1], [x-dx+1, y], [x-dx+1, y+dy-1]]
    # print(index)
    #cropped_img = (image[x:(x+dx), y:(y+dy),:], rgb_binary[x:(x+dx), y:(y+dy)], mask[x:(x+dx), y:(y+dy)])
    rgb_image1 = slide.read_region((x, y), 0, crop_size)
    rgb_mask1 = truth.read_region((x, y), 0, crop_size)
    rgb_mask1 = (cv2.cvtColor(np.array(rgb_mask1),
                              cv2.COLOR_RGB2GRAY) > 0).astype(int)

    rgb_image2 = slide.read_region((x, y+dy-1), 0, crop_size)
    rgb_mask2 = truth.read_region((x, y+dy-1), 0, crop_size)
    rgb_mask2 = (cv2.cvtColor(np.array(rgb_mask2),
                              cv2.COLOR_RGB2GRAY) > 0).astype(int)

    rgb_image3 = slide.read_region((x-dx+1, y), 0, crop_size)
    rgb_mask3 = truth.read_region((x-dx+1, y), 0, crop_size)
    rgb_mask3 = (cv2.cvtColor(np.array(rgb_mask3),
                              cv2.COLOR_RGB2GRAY) > 0).astype(int)

    rgb_image4 = slide.read_region((x-dx+1, y+dy-1), 0, crop_size)
    rgb_mask4 = truth.read_region((x-dx+1, y+dy-1), 0, crop_size)
    rgb_mask4 = (cv2.cvtColor(np.array(rgb_mask4),
                              cv2.COLOR_RGB2GRAY) > 0).astype(int)

    # rgb_mask = (cv2.cvtColor(np.array(rgb_mask),
    #                         cv2.COLOR_RGB2GRAY) > 0).astype(int)
    #cropped_img = image[x:(x+dx), y:(y+dy),:]
    #cropped_binary = rgb_binary[x:(x+dx), y:(y+dy)]
    #cropped_mask = mask[x:(x+dx), y:(y+dy)]
    print(index)
    return (rgb_image1, rgb_image2, rgb_image3, rgb_image4, rgb_mask1, rgb_mask2, rgb_mask3, rgb_mask4, index)


#sampletotal = pd.DataFrame([])
crop_size = [224, 224]
i = 0
while i < len(slide_paths):
    #sampletotal = pd.DataFrame([])
    base_truth_dir = Path(BASE_TRUTH_DIR)
    anno_path = Path(anno_path)

    with openslide.open_slide(slide_paths_total[i]) as slide:

        truth_slide_path = base_truth_dir / \
            osp.basename(slide_paths_total[i]).replace('.tif', '_mask.tif')
        Anno_pathxml = anno_path / \
            osp.basename(slide_paths_total[i]).replace('.tif', '.xml')

        with openslide.open_slide(str(truth_slide_path)) as truth:

            #slide = openslide.open_slide(slide_paths_total[i])
            annotations = convert_xml_df(str(Anno_pathxml))
            x_values = list(annotations['X'].get_values())
            y_values = list(annotations['Y'].get_values())
            bbox = zip(x_values, y_values)
            bbox = list(bbox)

            m = 0
            while m in range(0, len(bbox)):
                r = random_crop(slide, truth, crop_size, bbox[m])
                for n in range(0, 4):
                    if (cv2.countNonZero(r[n+4]) == 0):

                        saveim('/Users/liw17/Documents/new pred/%s_%d_%d.png' %
                               (osp.splitext(osp.basename(slide_paths_total[i]))[0], r[8][n][0], r[8][n][1]), r[n])

                        io.imsave('/Users/liw17/Documents/new pred/%s_%d_%d_mask.png' % (osp.splitext(
                            osp.basename(slide_paths_total[i]))[0], r[8][n][0], r[8][n][1]), r[n+4])

                        print(r[n])

                    m = m+5

                print(m)

    i = i+1
