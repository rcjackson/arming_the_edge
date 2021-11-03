from matplotlib import use
use("agg")
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import Cropping2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tfrecords_path = '/home/rjackson/tfrecords/2006/*'

is_training = True
shuffle = False

def input_fn():
    def parse_record(record):
        feature = {'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                   'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                   'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=""),
                   'time': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                   }
        features = tf.io.parse_single_example(record, feature)
        my_shape = (features['width'], features['height'], 1)
        features['image_raw'] = tf.io.decode_raw(features['image_raw'], tf.float64)
        features['image_raw'] = tf.reshape(features['image_raw'], shape=list(my_shape))

        return {'input_1': features['image_raw'], 'width': features['width'], 'height': features['height'],
                'conv2d_6': features['image_raw']}

    file_list = sorted(glob(tfrecords_path))
    dataset = tf.data.TFRecordDataset(file_list)

    if is_training:
        if shuffle:
            dataset = dataset.shuffle()
        else:
            dataset = dataset.repeat()

    dataset = dataset.map(parse_record)
    dataset = dataset.batch(20)
    dataset = tf.data.Dataset.zip((dataset, dataset))

    return dataset

def _int64_feature(value):
    """Creates a tf.Train.Feature from an int64 value."""
    if value is None:
        value = []
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def encoder_decoder_model():
    inp = Input(shape=(128, 256, 3), name="input")
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer='l2')(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer='l2')(x)
    encoded = MaxPooling2D((2, 2), name="encoding")(x)
    
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, kernel_size=(3, 3), padding='same', activation='relu', name="decoding")(x)
    return inp, encoded, decoded


if __name__ == "__main__":
    hidden_size = 5
    use_dropout = True
    num_steps = 3
    
    Gen = ImageDataGenerator(rescale=1/255.)
    dataset = Gen.flow_from_directory(
        '/lambda_stor/data/rjackson/lidar_pngs/5min_snr/training', class_mode='input', target_size=(256, 128), shuffle=False)
    
    #inp, encoder, decoded = encoder_decoder_model()
    model = load_model('/homes/rjackson/encoder/encoder-decoder-100.hdf5')
    encoder = Model(model.get_layer("input"), model.get_layer("encoding").output)
    out_images = model.predict(dataset)
     
    
    
