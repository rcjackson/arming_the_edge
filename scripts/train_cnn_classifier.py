import tensorflow as tf
import numpy as np

from glob import glob
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Add, Activation, Concatenate, Flatten
from tensorflow.keras.layers import Cropping2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.initalizers import VarianceScaling
tfrecords_path_snr = '/lambda_stor/data/rjackson/lidar_tfrecords/10min/*.tfrecord'
#tfrecords_path_vel = '/lambda_stor/data/rjackson/lidar_tfrecords/10min/velocity/*.tfrecord'

is_training = True
shuffle = False

def input_fn():
    def parse_record(record):
        feature = {'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                   'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                   'snr_image': tf.io.FixedLenFeature([], tf.string, default_value=""),
                   'vel_image': tf.io.FixedLenFeature([], tf.string, default_value=""),
                   'time': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                   'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)}
        features = tf.io.parse_single_example(record, feature)
        my_shape = (features['width'], features['height'], 1)
        features['snr_image'] = tf.io.decode_raw(features['snr_image'], tf.float64)
        features['snr_image'] = tf.reshape(features['snr_image'], shape=list(my_shape))
        features['vel_image'] = tf.io.decode_raw(features['vel_image'], tf.float64)
        features['vel_image'] = tf.reshape(features['vel_image'], shape=list(my_shape))

        return {'snr': features['snr_image'], 'width': features['width'], 'height': features['height'],
                'class': features['label'], 'velocity': features['vel_image']}

    def make_one_hot(record):
        record['label'] = tf.one_hot(record['class'], depth=3)
        return record

    def flip_lr(record):
        record['velocity'] = tf.image.flip_up_down(record['velocity'])
        record['snr'] = tf.image.flip_up_down(record['snr'])
        return record

    def normalize(record):
        record['velocity' ] = tf.image.per_image_standardization(record['velocity'])
        record['snr'] = tf.image.per_image_standardization(record['snr'])
        return record

    file_list = sorted(glob(tfrecords_path_snr))
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(parse_record)
    dataset = dataset.map(make_one_hot)
    dataset = dataset.map(normalize)
    #dataset = dataset.concatenate(dataset.map(flip_lr))
    dataset = dataset.shuffle(200)
    dataset = dataset.batch(32)
    dataset = tf.data.Dataset.zip((dataset, dataset))

    return dataset

def _int64_feature(value):
    """Creates a tf.Train.Feature from an int64 value."""
    if value is None:
        value = []
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Creates a tf.Train.Feature from a bytes value."""
    if value is None:
        value = []
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)
        value = bytes(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Creates a tf.Train.Feature from a bytes value."""
    if value is None:
        value = []
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)
        value = bytes(value)
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(axis=-1)(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(axis=-1)(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, input_name, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 2
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape, name=input_name)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = Add()([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    #outputs = Dense(num_classes,
    #                activation='softmax',
    #                kernel_initializer='he_normal')(y)

    # Instantiate model.
    #model = Model(inputs=inputs, outputs=outputs)
    return inputs, x


def conv_net_layer(inp, skip=False, num_channels=2, batch_norm=True,
                   activate=True):
    x = Conv2D(num_channels, kernel_size=(3, 3), kernel_initializer='he_normal')(inp)
    if batch_norm:
        x = BatchNormalization(axis=-1)(x)
    if activate:
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    return x


def conv_net_classifier(ds, velocity=False):
    wid = ds[0]['width'].numpy()[0]
    hei = ds[0]['height'].numpy()[0]
    print(wid, hei)
    ref_inp = Input(shape=(wid, hei, 1), name='snr')
    #ref_inp = BatchNormalization()(ref_inp)
    #vel_inp = Input(shape=(wid, hei, 1), name='velocity')

    pad_x = np.ceil(ds[0]['width'].numpy()[0] / 8) * 8 - ds[0]['width'].numpy()[0]
    pad_y = np.ceil(ds[0]['height'].numpy()[0] / 8) * 8 - ds[0]['height'].numpy()[0]
    ref = ZeroPadding2D(((0, int(pad_x)), (0, int(pad_y))))(ref_inp)
    
    #ref_in, ref_out = resnet_v2(input_shape=(wid, hei, 1), input_name='snr',depth=11, num_classes=3)
     
    layer2 = conv_net_layer(layer1, num_channels=32, batch_norm=False,
             activate=True)
    
    layer2 = conv_net_layer(layer2, num_channels=32, batch_norm=False,
             activate=False)

    layer2 = conv_net_layer(layer2, num_channels=32, batch_norm=False,
             activate=False)

    layer2 = conv_net_layer(layer2, num_channels=32, batch_norm=False,
             activate=False)
    ref_out = conv_net_layer(layer2, num_channels=1)

    #ref_out = conv_net_layer(layer2, num_channels=1, batch_norm=True, activate=False)
    #ref_skip = conv_net_layer(ref, num_channels=1, batch_norm=True, activate=False)
    #ref_skip = Activation('relu')(ref_skip)
    #ref_out = Add()([ref_out, ref_skip])
    #ref_out = Activation('relu')(ref_out)
    if velocity:   
        x = Concatenate()([ref_out, vel_out])
    else:
        x = ref_out 
    
    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = Flatten()(x)
    #x = AveragePooling2D()(x)
    outputs = Dense(1, activation='relu')(x)
    outputs = Dense(3, name='label',
                    activation='softmax',
                    kernel_initializer='he_normal')(outputs)

    #x = Dense(2, activation='relu')(x)
    #x = Dense(3, activation='softmax', name='label')(x)
    if velocity:
        return Model(inputs=[ref_in, vel_in], outputs=outputs)
    else:
        return Model(ref_inp, outputs)


if __name__ == "__main__":
    hidden_size = 5
    use_dropout = True
    num_steps = 3
    
    epoch_no = 0
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
           # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have bee initialized
            print(e)
   
    dataset = input_fn()
    testset = dataset.take(60)
    trainset = dataset.skip(60)
    model = conv_net_classifier([x[0] for x in trainset])
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    dataset = input_fn()
    checkpointer = ModelCheckpoint(
        filepath=('/homes/rjackson/arming_the_edge/models/classifier-%dframes-{epoch:03d}.hdf5'),
        verbose=1)
    model.fit(trainset, None, validation_data=testset, validation_steps=30, epochs=300, callbacks=[checkpointer], initial_epoch=epoch_no)
