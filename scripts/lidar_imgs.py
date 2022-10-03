import numpy as np
import xarray as xr
import time
import highiq
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import act
import sys

from distributed import Client, wait
from dask_cuda import LocalCUDACluster
from glob import glob

lidar_path = '/lcrc/group/earthscience/rjackson/sgpdlacfC1.a1/'
out_img_path = '/lcrc/group/earthscience/rjackson/sgp_lidar_imgs/C1/'
out_ds_path = '/lcrc/group/earthscience/rjackson/sgp_lidar/'
include_spectra = True
stare_list = sorted(glob(lidar_path + '*%s*.nc' % sys.argv[1]))

def load_file(file_path):
    test_file = highiq.io.load_arm_netcdf(file_path)
    return test_file

def process_file(ds):
    print("Processing lidar moments...")
    ti = time.time()
    my_list = []
    for x in ds.groupby_bins('time', ds.time.values[::600]):
        d = x[1]
        d['acf_bkg'] = d['acf_bkg'].isel(time=1)
        psd = highiq.calc.get_psd(d)
        my_list.append(highiq.calc.get_lidar_moments(psd))
        del psd
    ds_out = xr.concat(my_list, dim='time')
    ds_out = highiq.calc.calc_num_peaks(ds_out)
    # Just include moments in output
    if include_spectra is False:
        ds_out = ds_out.drop_dims(["vel_bin_interp", "nsamples", 
            "complex", "nlags"])
    else:
        ds_out = ds_out.drop_dims(["nlags", "complex", "nsamples"])
    print("Done in %3.2f minutes" % ((time.time() - ti) / 60.))
    return ds_out

def plot_and_process(fi):
    path, name = os.path.split(fi)
    splits = fi.split(".")
    ds = load_file(fi)
    ds = process_file(ds)

    ds['doppler_velocity'] = ds['doppler_velocity'].where(ds.snr >= 1)
    ds['vel_variance'] = (ds['doppler_velocity'] - ds['doppler_velocity'].mean(axis=0)).rolling(time=200).var()
    ds['vel_variance'].attrs["long_name"] = "Vertical velocity variance"
    ds['vel_variance'].attrs["units"] = "m-2 s-2"
    ds.to_netcdf(out_ds_path + '/processed_moments/' + name[:-6] + '.moments.nc')
    #if ds.time.values[-1] - ds.time.values[0] < np.datetime64(20, 'm'):
    #    ds.close()
    #    return
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ds['snr'] = ds['snr'].where(ds.snr >= 1)
    ds['snr'].T.plot(ax=ax, cmap='act_HomeyerRainbow',
            vmin=1, vmax=30)
    ax.set_ylim([0, 10000])
    fig.savefig(out_img_path + '/snr/'+ name + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ds['doppler_velocity'] = ds['doppler_velocity'].where(ds.snr >= 1)
    ds['doppler_velocity'].T.plot(ax=ax, cmap='act_balance',
          vmin=-10, vmax=10)
    ax.set_ylim([0, 10000])
    fig.savefig(out_img_path + '/vel/'+ name + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ds['skewness'] = ds['skewness'].where(ds.snr >= 1)
    ds['skewness'].T.plot(ax=ax, cmap='act_balance',
            vmin=-5, vmax=5)
    ax.set_ylim([0, 10000])
    fig.savefig(out_img_path + '/skewness/'+ name + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ds['kurtosis'] = ds['kurtosis'].where(ds.snr >= 1)
    ds['kurtosis'].T.plot(ax=ax, cmap='act_HomeyerRainbow',
            vmin=-20, vmax=20)
    ax.set_ylim([0, 10000])
    fig.savefig(out_img_path + '/kurtosis/'+ name + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ds['spectral_width'] = ds['spectral_width'].where(ds.snr >= 1)
    ds['spectral_width'].T.plot(ax=ax, cmap='act_HomeyerRainbow',
            vmin=0, vmax=10)
    ax.set_ylim([0, 10000])
    fig.savefig(out_img_path + '/spectral_width/'+ splits[-3] + '.' + splits[-2] + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    #with Client(LocalCUDACluster(n_workers=2)) as cli:
    #    results = cli.map(plot_and_process, stare_list)
    #    wait(results)
    for fi in stare_list:
        try:
            plot_and_process(fi)
        except TypeError:
            continue
