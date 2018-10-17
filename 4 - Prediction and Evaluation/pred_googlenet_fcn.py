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
import gc

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
import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.initializers import Constant

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights


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
image_size = (224, 224)

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
        img = slide.read_region(xylarge, 0, crop_size)
        img = np.array(img)
        img = img[:, :, :3]

        images.append(img)

    X_train = np.array(images)

    yield X_train


# -*- coding: utf-8 -*-
"""Inception V1 model for Keras.
Note that the input preprocessing function is different from the the VGG16 and ResNet models (same as Xception).
Also that (currently) the output predictions are for 1001 classes (with the 0 class being 'background'), 
so require a shift compared to the other models here.
# Reference
- [Going deeper with convolutions](http://arxiv.org/abs/1409.4842v1)
"""
import warnings
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing import image
from keras.regularizers import l2
WEIGHTS_PATH = ''
WEIGHTS_PATH_NO_TOP = ''
# conv2d_bn is similar to (but updated from) inception_v3 version
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              normalizer= False,
              activation='relu',
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution, `name + '_bn'` for the
            batch norm layer and `name + '_act'` for the
            activation layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'
    else:
        conv_name = None
        bn_name = None
        act_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
            filters, (num_row, num_col),
            strides=strides, padding=padding,
            use_bias=False, name=conv_name, kernel_regularizer=l2(0.0005))(x)
    if normalizer:
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation:
        x = Activation(activation, name=act_name)(x)
    return x
    
# Convenience function for 'standard' Inception concatenated blocks
def concatenated_block(x, specs, channel_axis, name):
    (br0, br1, br2, br3) = specs   # ((64,), (96,128), (16,32), (32,))
    
    branch_0 = conv2d_bn(x, br0[0], 1, 1, name=name+"_Branch_0_a_1x1")
    branch_1 = conv2d_bn(x, br1[0], 1, 1, name=name+"_Branch_1_a_1x1")
    branch_1 = conv2d_bn(branch_1, br1[1], 3, 3, name=name+"_Branch_1_b_3x3")
    branch_2 = conv2d_bn(x, br2[0], 1, 1, name=name+"_Branch_2_a_1x1")
    branch_2 = conv2d_bn(branch_2, br2[1], 3, 3, name=name+"_Branch_2_b_3x3")
    branch_3 = MaxPooling2D( (3, 3), strides=(1, 1), padding='same', name=name+"_Branch_3_a_max")(x)  
    branch_3 = conv2d_bn(branch_3, br3[0], 1, 1, name=name+"_Branch_3_b_1x1")
    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name=name+"_Concatenated")
    return x
def InceptionV1(include_top=True,
                #weights= None,
                weights= 'imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=2):
    """Instantiates the Inception v1 architecture.
    This architecture is defined in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/abs/1409.4842v1
    
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 224x224.
    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
   # if weights == 'imagenet' and include_top and classes != 1001:
   #     raise ValueError('If using `weights` as imagenet with `include_top`'
   #                      ' as true, `classes` should be 1001')
    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        #default_size=299,
        default_size=224,
        min_size=139,
        data_format=K.image_data_format(),
        include_top=include_top)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    # 'Sequential bit at start'
    x = img_input
    x = conv2d_bn(x,  64, 7, 7, strides=(2, 2), padding='same',  name='Conv2d_1a_7x7')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_2a_3x3')(x)  
    
    x = conv2d_bn(x,  64, 1, 1, strides=(1, 1), padding='same', name='Conv2d_2b_1x1')  
    x = conv2d_bn(x, 192, 3, 3, strides=(1, 1), padding='same', name='Conv2d_2c_3x3')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_3a_3x3')(x)  
    
    # Now the '3' level inception units
    x = concatenated_block(x, (( 64,), ( 96,128), (16, 32), ( 32,)), channel_axis, 'Mixed_3b')
    x = concatenated_block(x, ((128,), (128,192), (32, 96), ( 64,)), channel_axis, 'Mixed_3c')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_4a_3x3')(x)  
    # Now the '4' level inception units
    x = concatenated_block(x, ((192,), ( 96,208), (16, 48), ( 64,)), channel_axis, 'Mixed_4b')
    x = concatenated_block(x, ((160,), (112,224), (24, 64), ( 64,)), channel_axis, 'Mixed_4c')
    x = concatenated_block(x, ((128,), (128,256), (24, 64), ( 64,)), channel_axis, 'Mixed_4d')
    x = concatenated_block(x, ((112,), (144,288), (32, 64), ( 64,)), channel_axis, 'Mixed_4e')
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_4f')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='MaxPool_5a_2x2')(x)  
    # Now the '5' level inception units
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_5b')
    x = concatenated_block(x, ((384,), (192,384), (48,128), (128,)), channel_axis, 'Mixed_5c')
    
    if include_top:
        # Classification block
        
        # 'AvgPool_0a_7x7'
        x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)  
        
        # 'Dropout_0b'
        x = Dropout(0.5)(x)  # slim has keep_prob (@0.8), keras uses drop_fraction
        
        #logits = conv2d_bn(x,  classes+1, 1, 1, strides=(1, 1), padding='valid', name='Logits',
        #                   normalizer=False, activation=None, )  
        
        # Write out the logits explictly, since it is pretty different
        x = Conv2D(classes, (1, 1), strides=(1,1), padding='valid', use_bias=True, name='Logits')(x)
        
        #x = Flatten(name='Logits_flat')(x)
        #print(x.get_shape().as_list())
        #x = x[:, 1:]  # ??Shift up so that first class ('blank background') vanishes
        # Would be more efficient to strip off position[0] from the weights+bias terms directly in 'Logits'
        #x = Conv2DTranspose(2, (64, 64), strides=(32, 32), filters=x.get_shape().as_list(), padding='same')

        #x = Conv2DTranspose(2, (64, 64), strides=(32, 32), activation='softmax', padding='same')
        x = BilinearUpSampling2D(target_size=tuple(image_size))(x)
        #x = Conv2D(filters=2, 
        #       kernel_size=(1, 1))(x)
        #x = Conv2DTranspose(filters=2, 
        #                 kernel_size=(64, 64),
        #                strides=(32, 32),
        #                padding='same',
        #                activation='softmax',
                        #kernel_initializer=Constant(bilinear_upsample_weights(32, 2))
        #                 )(x)

        #x = Activation('softmax', name='Predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_pooling')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='global_pooling')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Finally : Create model
    model = Model(inputs, x, name='inception_v1')
    
    # LOAD model weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path ='/home/wli/Downloads/googlenet0917-02-0.93.hdf5'

            #get_file(
            #    'inception_v1_weights_tf_dim_ordering_tf_kernels.h5',
            #    WEIGHTS_PATH,
            #    cache_subdir='models',
            #    md5_hash='723bf2f662a5c07db50d28c8d35b626d')
        else:

            weights_path ='/home/wli/Downloads/googlenet0917-02-0.93.hdf5'
            #weights_path = get_file(
            #    'inception_v1_weights_tf_dim_ordering_tf_kernels_notop.h5',
            #   WEIGHTS_PATH_NO_TOP,
            #    cache_subdir='models',
            #    md5_hash='6fa8ecdc5f6c402a59909437f0f5c975')
        model.load_weights('/home/wli/Downloads/googlenet0917-02-0.93.hdf5')
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)    
    
    return model


def fcn_32s():
    inputs = Input(shape=(None, None, 3))
    googlenet = InceptionV1(weights='imagenet', include_top=True, input_tensor=inputs)
    x = Conv2D(filters=nb_classes, 
               kernel_size=(1, 1))(googlenet.output)
    x = Conv2DTranspose(filters=nb_classes, 
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(x)
    model = Model(inputs=inputs, outputs=x)
    #for layer in model.layers[:15]:
    #    layer.trainable = False
    return model


model =InceptionV1()
print("model built")
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.00001, decay=0, momentum=0, nesterov=False)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
print("model compiled")

#filepath="\home\wli\Downloads\googlenet1017model2-{epoch:02d}-{val_acc:.2f}.hdf5"


#model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)


def predict_batch_from_model(patches, model):

    predictions = model.predict(patches)
    #print(predictions[:, 1])
    #print(predictions[:, 0])
    predictions = predictions[:,:,:,1]
    print(predictions)
    return predictions


#model = load_model(
#    '/home/wli/Downloads/googlenet0917-02-0.93.hdf5')
#alpha = 0.5

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

        output_thumbnail_preds.extend(preds)
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
        del preds
        gc.collect()
    np.save('/home/wli/Downloads/googlepred/%s' %
            (osp.splitext(osp.basename(slide_paths[i]))[0]), output_thumbnail_preds_fcn)

    i = i+1
