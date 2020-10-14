"""
This script will convert the lidar images to TFRecords with given time intervals.

This converts the data to greyscale images by:
0 = all masked values
1 dB = 0.1, 30 dB = 1
0.1 velocity = -19.35 nyquist
1.0 velocity =
"""
import sys
import numpy as np
import xarray as xr
import tensorflow as tf
import os
import glob
import pandas as pd

from datetime import datetime

tfrecords_path = '/lambda_stor/data/rjackson/lidar_tfrecords/velocity/'
lidar_files = "/lambda_stor/data/rjackson/sgp_lidar/*moments*.nc"
first_shape = None
label_df = pd.read_csv('../notebooks/lidar_labels.csv')

date_list = np.array([datetime.strptime(x, '%Y-%m-%d').date() for x in label_df["Date"].values])
start_time_list = np.array([datetime.strptime(x[0:4], '%H%M').time() for x in label_df["Time"].values])
end_time_list = np.array([datetime.strptime(x[5:], '%H%M').time() for x in label_df["Time"].values])

def get_label(dt):
    label_ind = np.where(np.logical_and.reduce(
        (date_list == dt.date(), start_time_list <= dt.time(), end_time_list > dt.time())))
    if not label_ind[0].size:
        return -1
    my_string = label_df["Label"].values[label_ind][0]

    if my_string.lower() == 'clear':
        return 0
    elif my_string.lower() == 'cloudy' or my_string.lower() == "cloud":
        return 1
    elif my_string.lower() == 'rain':
        return 2

    raise ValueError("Invalid value %s for label" % my_string)


def dt64_to_dt(dt):
    ts = (dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)

def create_tf_record(file, time_interval=5, first_shape=None):
    # Get radar file from bebop

    grid = xr.open_dataset(file)
    Zn = grid.snr.where(grid.snr > 1).fillna(0).values
    Zn = Zn/30
    Zn = np.where(Zn > 1, 1, Zn)

    times = grid.time.values
    grid.close()
    shp = Zn.shape
    cur_time = times[0]
    end_time = times[-1]

    start_ind = 0
    i = 0
    while cur_time < end_time:
        next_time = cur_time + np.timedelta64(time_interval, 'm')
        if next_time > end_time:
            next_ind = len(times)
        else:
            next_ind = np.argmin(np.abs(next_time - times))

        my_data = Zn[start_ind:next_ind, :]
        my_times = times[start_ind:next_ind]
        if len(my_times) == 0:
            break
        cur_time = next_time
        start_ind += next_ind - start_ind + 1

        if first_shape is None:
            first_shape = my_data.shape
        else:
            if my_data.shape[0] > first_shape[0]:
                my_data = my_data[:first_shape[0], :]
            elif my_data.shape[0] < first_shape[0]:
                my_data = np.pad(my_data, [(0, first_shape[0]-my_data.shape[0]), (0, 0)],
                                 mode='constant')
        dir_path = '%dmin/' % time_interval
        if not os.path.exists(tfrecords_path + dir_path):
            os.makedirs(tfrecords_path + dir_path)

        fname = tfrecords_path + dir_path + file.split("/")[-1][:-3] + '%d.tfrecord' % i
        print(fname)
        writer = tf.io.TFRecordWriter(fname)
        width = first_shape[0]
        height = first_shape[1]
        # norm = norm.SerializeToString()

        label = get_label(dt64_to_dt(my_times[0]))
        if label > -1:
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'width': _int64_feature(width),
                    'height': _int64_feature(height),
                    'image_raw': _bytes_feature(my_data),
                    'start_time': _float_feature(my_times[0]),
                    'end_time': _float_feature(my_times[-1]),
                    'label': _int64_feature(label)
                }))
            writer.write(example.SerializeToString())
        i += 1
    return first_shape


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

my_files = glob.glob(lidar_files)
for file in my_files:
    first_shape = create_tf_record(file, 10, first_shape)
