import tensorflow as tf
import numpy as np

from glob import glob
from keras_unet.models import custom_unet
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Add, Activation, Concatenate, Flatten
from tensorflow.keras.layers import Cropping2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, Dropout, Reshape
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from PIL import Image
from random import shuffle

import matplotlib.image as mpimg
import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"  


cloudy_data_path = '/lambda_stor/data/rjackson/lidar_pngs/5min_snr/training/rainy/*.png'
cloud_images = glob(cloudy_data_path)

train_datagen_flip = ImageDataGenerator(rescale=1/255.,
                                        horizontal_flip=True,
                                        width_shift_range=[-5, 5])
j = 0
for image in cloud_images:
    img = load_img(image)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in train_datagen_flip.flow(x, batch_size=1,
                                        save_to_dir='/lambda_stor/data/rjackson/lidar_pngs/augmented/rainy/',
                                        save_prefix=str(j) + str(i), save_format='png'):
        i += 1
        if i > 15:
            break

    j += 1
    if j % 100 == 0:
        print('%d/%d' % (j, len(cloud_images)))
