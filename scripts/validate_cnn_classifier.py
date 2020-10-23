import tensorflow as tf
import numpy as np
import sys

from glob import glob
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Activation, Add
from tensorflow.keras.layers import Cropping2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint

tfrecords_path = '/lambda_stor/data/rjackson/lidar_tfrecords/10min/*.tfrecord'


is_training = True
shuffle = False

def input_fn():
    def parse_record(record):
        feature = {'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                   'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                   'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=""),
                   'time': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                   'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)}
        features = tf.io.parse_single_example(record, feature)
        my_shape = (features['width'], features['height'], 1)
        features['image_raw'] = tf.io.decode_raw(features['image_raw'], tf.float64)
        features['image_raw'] = tf.reshape(features['image_raw'], shape=list(my_shape))

        return {'input': features['image_raw'], 'width': features['width'], 'height': features['height'],
                'class': features['label']}

    def make_one_hot(record):
        record['label'] = tf.one_hot(record['class'], depth=3)
        return record 
    file_list = sorted(glob(tfrecords_path))
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(parse_record)
    dataset = dataset.map(make_one_hot)
    dataset = dataset.shuffle(150)
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


def conv_net_classifier(ds):
    wid = ds[0]['width'].numpy()[0]
    hei = ds[0]['height'].numpy()[0]
    inp = Input(shape=(wid, hei, 1), name='input')
    
    pad_x = np.ceil(ds[0]['width'].numpy()[0] / 8) * 8 - ds[0]['width'].numpy()[0]
    pad_y = np.ceil(ds[0]['height'].numpy()[0] / 8) * 8 - ds[0]['height'].numpy()[0]
    x = ZeroPadding2D(((0, int(pad_x)), (0, int(pad_y))))(inp)
    x = Conv2D(2, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(4, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1, activation='relu')(x)
    x = Dense(3, activation='softmax', name='label')(x)
    return inp, x


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
            tf.config.experimental.set_visible_devices(gpus[4], 'GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have bee initialized
            print(e)    
    
     
    dataset = input_fn()
    trainset = dataset.skip(50)
    testset = dataset.take(50)
    inp, out = conv_net_classifier([x[0] for x in dataset])
    model = load_model(sys.argv[1])
    output = model.predict(testset, steps=50)
    labels = np.stack([x[0]['class'] for x in testset]).flatten()
    probs = np.argmax(output, axis=1).flatten()

    num_correct = len(np.where(np.equal(labels, probs))[0])
    print("percent correct labels: %3.2f" % (num_correct/len(labels)*100))


