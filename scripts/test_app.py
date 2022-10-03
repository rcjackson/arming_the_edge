import numpy as np
import xgboost as xgb
import xarray as xr
import pandas as pd
import highiq
import time
import argparse
import os
import base64
import json
import xarray as xr
import io

from glob import glob

def open_load_model(model_path):
    print(model_path)
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst

def load_file(file_path):
    test_file = highiq.io.read_00_data(file_path, 'sgpdlprofcalE37.home_point')
    test_file.to_netcdf('test.nc')
    del test_file
    test_file = xr.open_dataset('test.nc')
    return test_file

def process_file(ds):
    print("Processing lidar moments...")
    ti = time.time()
    my_list = []
    for x in ds.groupby_bins('time', ds.time.values[::5]):
        d = x[1]
        d['acf_bkg'] = d['acf_bkg'].isel(time=1)
        psd = highiq.calc.get_psd(d)
        my_list.append(highiq.calc.get_lidar_moments(psd))
        del psd
    ds_out = xr.concat(my_list, dim='time')

    print("Done in %3.2f minutes" % ((time.time() - ti) / 60.))
    return ds_out


def get_scp(ds, model_name, config):
    range_bins = np.arange(60., 11280., 120.)
    # Parse model string for locations of snr, mean_velocity, spectral_width
    locs = 0
    snr_thresholds = []
    scp_ds = {}
    if config == "Stare":
        interval = 5
        dates = pd.date_range(ds.time.values[0],
                          ds.time.values[-1], freq='%dmin' % interval)
    else:
        dates = pd.date_range(ds.time.values[0],
                          ds.time.values[-1], periods=2)
    times = ds.time.values
    snr = ds['snr'].values
    mname = model_name
    while locs > -1:
        locs = mname.find("snr")
        if locs > -1:
            snr_thresholds.append(float(mname[locs+5:locs+13]))
            scp_ds['snrgt%f' % snr_thresholds[-1]] = np.zeros(
                    (len(dates) - 1, len(range_bins) - 1))
            mname = mname[locs+13:]
    
    for i in range(len(dates) - 1):
        time_inds = np.argwhere(np.logical_and(ds.time.values >= dates[i],
                                               ds.time.values < dates[i + 1]))
        if len(time_inds) == 0:
            continue
        for j in range(len(range_bins) - 1):
            range_inds = np.argwhere(np.logical_and(
                ds.range.values >= range_bins[j], 
                ds.range.values < range_bins[j+1]))
            range_inds = range_inds.astype(int)
            if len(range_inds) == 0:
                continue
            snrs = snr[int(time_inds[0]):int(time_inds[-1]), 
                    int(range_inds[0]):int(range_inds[-1])]
            for snr_thresh in snr_thresholds:
                scp_ds['snrgt%f' % snr_thresh][i, j] = len(np.argwhere(snrs > snr_thresh)) / \
                                                       (len(time_inds) * len(range_inds)) * 100
    scp_ds['input_array'] = np.concatenate(
            [scp_ds[var_keys] for var_keys in scp_ds.keys()], axis=1)
    scp_ds['time_bins'] = dates
    print(scp_ds['input_array'].shape)
    return scp_ds


def progress(bytes_so_far: int, total_bytes: int):
    pct_complete = 100. * float(bytes_so_far) / float(total_bytes)
    if int(pct_complete * 10) % 100 == 0:
        print("Total progress = %4.2f" % pct_complete)  


def worker_main(args):
    print('opening input %s' % args.input)
    file_list = glob(args.input + '/*%s*.raw' % args.config)
    class_names = ['clear', 'cloudy', 'rain']
    
    predicts = []
    times = []
    model = open_load_model(args.model)
    for file_name in file_list:
        print("Processing %s" % file_name)
        dsd_ds = load_file(file_name)
        dsd_ds = process_file(dsd_ds)
        scp = get_scp(dsd_ds, args.model, args.config)
        out_predict = model.predict(xgb.DMatrix(scp['input_array']))
        for i in range(len(out_predict)):
            print(scp['time_bins'][i].timestamp())
            print(str(
               scp['time_bins'][i]) + ':' + class_names[int(out_predict[i])])
            times.append(scp['time_bins'][i].timestamp())
            predicts.append(int(out_predict[i]))

        #if out_predict[i] > 0:
        #    out_ds = dsd_ds.sel(time=slice(
        #        str(scp['time_bins'][i]), str(scp['time_bins'][i+1])), method='nearest')
        #    t = pd.to_datetime(out_ds.time.values[0])
        #    out_ds.to_netcdf('%s.nc' % 
        #        t.strftime('%Y%m%d.%H%M%S'))
                 
        dsd_ds.close() 
    out_df = pd.DataFrame({'time': np.array(times), 'predicts': np.array(predicts)})
    out_df = out_df.sort_values('time')
    out_df = out_df.set_index('time')
    out_df.to_csv('%s_predictions.csv' % args.config)


def main(args):
    worker_main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', dest='input',
        action='store', 
        help='Path to input device or ARM datastream name')
    parser.add_argument(
            '--config', dest='config', action='store', default='Stare')
    parser.add_argument(
        '--model', dest='model',
        action='store', default='modelsnrgt3.000000snrgt5.000000.json',
        help='Path to model')

    main(parser.parse_args())
