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
#from dask_cuda import LocalCUDACluster
from glob import glob

lidar_path = '/lcrc/group/earthscience/rjackson/sgpdlacfC1.00/'
out_img_path = '/lcrc/group/earthscience/rjackson/sgp_lidar_imgs/C1/'
out_ds_path = '/lcrc/group/earthscience/rjackson/sgp_lidar/processed_moments/'

stare_list = sorted(glob(lidar_path + '*Stare*.raw'))
#stare_list = sorted(glob(out_ds_path + '*moments*.nc'))

def load_file(file_path):
    test_file = xr.open_dataset(file_path)
    print(test_file)
    return test_file

def plot_and_process(fi):
    base, name = os.path.split(fi)
    splits = name.split(".")
    ds = load_file(fi)
    if ds.time[-1].values - ds.time[0].values < np.timedelta64(20, 'm'):
        ds.close()
        return
    ds['mean_velocity'] = ds['mean_velocity'].where(ds.snr >= 1)
    #ds['vel_variance'] = (ds['doppler_velocity'] - ds['doppler_velocity'].mean(axis=0)).rolling(time=200).var()
    #ds['vel_variance'].attrs["long_name"] = "Vertical velocity variance"
    #ds['vel_variance'].attrs["units"] = "m-2 s-2"
    ds.to_netcdf(out_ds_path + splits[-3] + '.' + splits[-2] + '.nc')
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ds['snr'] = ds['snr'].where(ds.snr >= 1)
    ds['snr'].T.plot(ax=ax, cmap='act_HomeyerRainbow',
            vmin=1, vmax=30)
    ax.set_ylim([0, 4000])
    
    fig.savefig(out_img_path + '/snr/' + splits[-4] + '.' +
            splits[-3] + '.' + splits[-2] + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ds['mean_velocity'] = ds['mean_velocity'].where(ds.snr >= 1)
    ds['mean_velocity'].T.plot(ax=ax, cmap='act_balance',
          vmin=-10, vmax=10)
    ax.set_ylim([0, 4000])
    fig.savefig(out_img_path + '/vel/'+ splits[-4] + '.' + 
            splits[-3] + '.' + splits[-2] + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ds['skewness'] = ds['skewness'].where(ds.snr >= 1)
    ds['skewness'].T.plot(ax=ax, cmap='act_balance',
            vmin=-5, vmax=5)
    ax.set_ylim([0, 4000])
    fig.savefig(out_img_path + '/skewness/' + splits[-4] + '.' + 
            splits[-3] + '.' + splits[-2] + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    #fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    #ds['kurtosis'] = ds['kurtosis'].where(ds.snr >= 1)
    #ds['kurtosis'].T.plot(ax=ax, cmap='act_HomeyerRainbow',
    #        vmin=-20, vmax=20, add_colorbar=False)
    #ax.set_ylim([0, 4000])
    #ax.axis('off')
    #fig.savefig(out_img_path + '/kurtosis/'+ splits[-3] + '.' + splits[-2] + '.png',
    #            bbox_inches='tight', dpi=300)
    #plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ds['spectral_width'] = ds['spectral_width'].where(ds.snr >= 1)
    ds['spectral_width'].T.plot(ax=ax, cmap='coolwarm',
            vmin=2, vmax=8)
    ax.set_ylim([0, 5000])
    fig.savefig(out_img_path + '/spectral_width/' + splits[-4] + '.' + 
            splits[-3] + '.' + splits[-2] + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    #fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    #ds['npeaks'] = ds['npeaks'].where(ds.snr >= 1)
    #ds['npeaks'].T.plot(ax=ax, cmap='act_HomeyerRainbow',
    #        vmin=0, vmax=4, add_colorbar=False)
    #ax.set_ylim([0, 4000])
    #ax.axis('off')
    #fig.savefig(out_img_path + '/num_peaks/'+ splits[-3] + '.' + splits[-2] + '.png',
    #            bbox_inches='tight', dpi=300)
    #plt.close(fig)
    #fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    #ds['vel_variance'] = ds['vel_variance'].where(ds.snr >= 1)
    #ds['vel_variance'].T.plot(ax=ax, cmap='act_HomeyerRainbow',
    #        vmin=0, vmax=1, add_colorbar=False)
    #ax.set_ylim([0, 4000])
    #ax.axis('off')
    #fig.savefig(out_img_path + '/vel_variance/'+ splits[-3] + '.' + splits[-2] + '.png',
    #            bbox_inches='tight', dpi=300)
    #plt.close(fig)
    #fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    #img_array_mask = ds.snr.T.values < 1
    #print(np.nanmin(ds.snr.values), np.nanmax(ds.snr.values))
    #print(np.nanmin(ds.kurtosis.values), np.nanmax(ds.kurtosis.values))
    #print(np.nanmin(ds.spectral_width.values), np.nanmax(ds.spectral_width.values))
    #img_array = np.stack([(-ds.doppler_velocity.T.values)/2, 
    #    (ds.doppler_velocity.T.values)/2, 
    #    (5 - ds.spectral_width.T.values)/3], axis=-1)
    #img_array = np.ma.masked_invalid(img_array)
    #img_array[img_array > 1] = 1
    #for i in range(2):
    #    img_array[:, :, i] = np.ma.masked_where(img_array_mask, img_array[:, :, i])
    #img_array = img_array.filled(0.)
    #where_black = np.all(img_array <= 0, axis=-1)
    #img_array[where_black, 0] = 0.
    #img_array[where_black, 1] = 0.
    #img_array[where_black, 2] = 0.
    #ax.imshow(img_array[:100, :300, :])
    #ax.invert_yaxis()
    #ax.axis('off')
    #fig.savefig(out_img_path + '/skew_kurt_snr/'+ splits[-3] + '.' + splits[-2] + '_1.png',
    #            bbox_inches='tight', dpi=300)
    #ax.imshow(img_array[:100, 300:600, :])
    #ax.invert_yaxis()
    #ax.axis('off')
    #fig.savefig(out_img_path + '/skew_kurt_snr/'+ splits[-3] + '.' + splits[-2] + '_2.png',
    #            bbox_inches='tight', dpi=300)
    #ax.imshow(img_array[:100, 600:900, :])
    #ax.invert_yaxis()
    #ax.axis('off')
    #fig.savefig(out_img_path + '/skew_kurt_snr/'+ splits[-3] + '.' + splits[-2] + '_3.png',
    #            bbox_inches='tight', dpi=300)
    #ax.imshow(img_array[:100, 900:1200, :])
    #ax.invert_yaxis()
    #ax.axis('off')
    #fig.savefig(out_img_path + '/skew_kurt_snr/'+ splits[-3] + '.' + splits[-2] + '_4.png',
    #            bbox_inches='tight', dpi=300)

     
    ds.close()
    plt.close(fig)

if __name__ == "__main__":
    for fi in stare_list:
        plot_and_process(fi)
    #with Client(LocalCUDACluster(n_workers=2)) as cli:
        #results = cli.map(plot_and_process, stare_list)
        #wait(results)
