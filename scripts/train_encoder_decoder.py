import tensorflow as tf
from glob import glob
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Cropping2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model


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


def encoder_decoder_model(ds):
    inp = Input(shape=(None, None, 1))
    pad_x = np.ceil(ds[0][0]['width'].numpy()[0] / 8) * 8 - ds[0][0]['width'].numpy()[0]
    pad_y = np.ceil(ds[0][0]['height'].numpy()[0] / 8) * 8 - ds[0][0]['height'].numpy()[0]
    x = ZeroPadding2D(((0, int(pad_x)), (0, int(pad_y))))(inp)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, kernel_size=(3, 3), activation='sigmoid', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Cropping2D(((0, int(pad_x)), (0, int(pad_y))))(x)
    decoded = Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
    return inp, encoded, decoded


if __name__ == "__main__":
    hidden_size = 5
    use_dropout = True
    num_steps = 3
    the_shape = my_shape
    epoch_no = int(sys.argv[1])
    num_frames = int(sys.argv[2])

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with tf.Session() as sess:
        dataset = input_fn()
        trainset = dataset.shard(2, 0)
        testset = dataset.shard(2, 1)

        inp, encoder, model = encoder_decoder_model(trainset)
        model = Model(inp, decoded)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
        dataset = input_fn()
        checkpointer = ModelCheckpoint(
            filepath=('/home/rjackson/DNNmodel/model-%dframes-{epoch:03d}.hdf5' % num_frames),
            verbose=1)
        my_model.fit(dataset, None, epochs=300, steps_per_epoch=1600, callbacks=[checkpointer], initial_epoch=epoch_no)


