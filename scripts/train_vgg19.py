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

def vgg(IMG_HEIGHT=256, IMG_WIDTH=128):
    restnet = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
    restnet.summary()
    output = restnet.layers[-1].output
    output = Flatten()(output)
    restnet = Model(restnet.input,output)
    for layer in restnet.layers[:-4]:
        layer.trainable = False
    model = Sequential()
    model.add(restnet)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, name='targets',
                    activation='softmax',
                    kernel_initializer='he_normal'))
    return model

#model = vgg()
model = load_model('/homes/rjackson/arming_the_edge/models/vgg19-combined-1layer-052.hdf5')
model.compile(optimizer=Adam(lr=0.001), 
        loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
        directory='/lambda_stor/data/rjackson/lidar_pngs/augmented',
        class_mode='categorical', classes=['clear', 'cloudy', 'rain'],
        target_size=(256, 128), shuffle=True, batch_size=32)

valid_generator = train_datagen.flow_from_directory(
        directory='/lambda_stor/data/rjackson/lidar_pngs/5min_snr/validation',
        class_mode='categorical', classes=['clear', 'cloudy', 'rain'],
        target_size=(256, 128), shuffle=True, batch_size=32)

class_weight = {0: 1.,
                1: 3.,
                2: 5.}
checkpointer = ModelCheckpoint(
               filepath=('/homes/rjackson/arming_the_edge/models/vgg19-combined-1layer-{epoch:03d}.hdf5'),verbose=1)
early_stopping = EarlyStopping(restore_best_weights=True, patience=100,
        monitor="val_accuracy", mode="max")
history = model.fit(train_generator, validation_data=valid_generator,
        epochs=2000, callbacks=[checkpointer, early_stopping], initial_epoch=53)

