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

tfrecords_path = '/nfs/gce/projects/digr/lidar_tfrecords/'
lidar_files = "/run/user/7920/gvfs/sftp:host=bebop.lcrc.anl.gov/lcrc/group/earthscience/rjackson/sgp_lidar/processed_moments/sgpdlacfC1.a1*moments.nc"
first_shape = None

def create_tf_record(file, time_interval=5, first_shape=None):
    # Get radar file from bebop

    grid = xr.open_dataset(file)
    vel = grid.mean_velocity
    vel += 20
    vel = vel.fillna(0)
    vel = vel/(2*19.85)
    vel = np.where(vel > 1, 1, vel)

    times = grid.time.values
    grid.close()
    shp = vel.shape
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

        my_data = vel[start_ind:next_ind, :]
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
        dir_path = '%dmin_vel/' % time_interval
        if not os.path.exists(tfrecords_path + dir_path):
            os.makedirs(tfrecords_path + dir_path)

        fname = tfrecords_path + dir_path + file.split("/")[-1][:-3] + '%d.tfrecord' % i
        print(fname)
        writer = tf.io.TFRecordWriter(fname)
        width = first_shape[0]
        height = first_shape[1]
        # norm = norm.SerializeToString()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'width': _int64_feature(width),
                'height': _int64_feature(height),
                'image_raw': _bytes_feature(my_data),
                'start_time': _float_feature(my_times[0]),
                'end_time': _float_feature(my_times[-1])
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
    first_shape = create_tf_record(file, 5, first_shape)
