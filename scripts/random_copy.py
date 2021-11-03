import random
import os
import glob
import shutil

file_list = glob.glob('/lambda_stor/data/rjackson/lidar_pngs/5min_snr/training/clear/*.png')
out_dir = '/lambda_stor/data/rjackson/lidar_pngs/5min_undersample/training/clear/'
if not os.path.exists(out_dir):
    os.path.mkdirs(out_dir)
for f in file_list:
    if random.random() < 0.33:
        shutil.copy(f, out_dir)


