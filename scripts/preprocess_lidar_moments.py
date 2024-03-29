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
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings
import act

from datetime import datetime
warnings.filterwarnings("ignore")

tfrecords_path = '/lcrc/group/earthscience/rjackson/lidar_pngs/'
lidar_files = "/lcrc/group/earthscience/rjackson/sgp_lidar/processed_moments/*moments*.nc"
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
    Zn = grid.snr.where(grid.snr > 1).values
    times = grid.time.values
    ranges = grid.range.values
    which_ranges = np.where(ranges < 8000.)[0]
    grid.close()
    shp = Zn.shape
    cur_time = times[0]
    end_time = times[-1]
    
    start_ind = 0
    i = 0
    while cur_time < end_time:
        next_time = cur_time + np.timedelta64(time_interval, 'm')
        print((next_time, end_time))
        if next_time > end_time:
            next_ind = len(times)
        else:
            next_ind = np.argmin(np.abs(next_time - times))
        if(start_ind >= next_ind):
            break
        my_data = Zn[start_ind:next_ind, 0:which_ranges[-1]].T
        #my_vd_data = Vd[start_ind:next_ind, :].T
        #my_data = np.stack([
        #    my_data, my_data, my_data], axis=2)
        my_data = my_data
        #my_data = my_data[:, -1:0:-1, :]
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
        dir_path = '%dmin_snr/' % time_interval 
        label = get_label(dt64_to_dt(my_times[0]))
        if label == 0:
            lab = 'clear'
        elif label == 1:
            lab = 'cloudy'
        elif label == -1:
            continue
        else:
            lab = 'rain' 
        if random.random() <= 0.8:
            which = 'training'
        else:
            which = 'validation'
        dir_path = dir_path + '%s/%s/' % (which, lab)
        if not os.path.exists(tfrecords_path + dir_path):
            os.makedirs(tfrecords_path + dir_path)

        fname = tfrecords_path + dir_path + file.split("/")[-1][:-3] + '%d.png' % i
        width = first_shape[0]
        height = first_shape[1]
        # norm = norm.SerializeToStri
        fig, ax = plt.subplots(1, 1, figsize=(1*(height/width), 1))
        #ax.imshow(my_data)
        ax.pcolormesh(my_data, cmap='act_HomeyerRainbow', vmin=1, vmax=30)
        ax.set_axis_off()
        ax.margins(0, 0)
        try:
            fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
        except RuntimeError:
            plt.close(fig)
            continue
        plt.close(fig)
        i = i + 1
        del fig, ax
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
