import xarray as xr
import pandas as pd
import numpy as np

from glob import glob

file_list = glob('/lcrc/group/earthscience/rjackson/sgp_lidar/processed_moments/*moments.nc')

thresholds = [1., 3., 5., 10., 20.]
variables = ['snr', 'doppler_velocity']
out_path = '/lcrc/group/earthscience/rjackson/sgp_lidar/coverage_product/'

range_bins = np.arange(0., 12000., 60)
snr_bins = np.arange(1., 51, 1.)
vel_bins = np.arange(-20., 20., 0.5)

for f in file_list:
    print(f)
    splits = f.split('.')
    ymd = splits[2]
    hms = splits[3]
    inp_ds = xr.open_dataset(f)
    scp_ds = {}
    time_range = pd.date_range(inp_ds.time[0].values, inp_ds.time[-1].values, 20)
    for v in variables:
        for t in thresholds:
            arr = np.zeros((19, len(range_bins) - 1))
            hist_array = np.zeros((19, len(range_bins) - 1, len(snr_bins) - 1))
            hist_array_vel = np.zeros((19, len(range_bins) - 1, len(vel_bins) - 1))

            for i in range(19):
                time_inds = np.argwhere(np.logical_and(inp_ds.time.values >= time_range[i],
                    inp_ds.time.values < time_range[i+1]))
                
                hist_array[i] = np.squeeze(np.apply_along_axis(
                        lambda a: np.histogram(a, bins=snr_bins)[0],
                        0, inp_ds['snr'].values[time_inds, 1:-1:2])).T
                hist_array[i] = hist_array[i] + \
                        np.squeeze(np.apply_along_axis(
                        lambda a: np.histogram(a, bins=snr_bins)[0],
                        0, inp_ds['snr'].values[time_inds, 0:-2:2])).T
                hist_array_vel[i] = np.squeeze(np.apply_along_axis(
                        lambda a: np.histogram(a, bins=vel_bins)[0],
                        0, inp_ds['doppler_velocity'].values[time_inds, 1:-1:2])).T
                hist_array_vel[i] = hist_array_vel[i] + \
                        np.squeeze(np.apply_along_axis(
                        lambda a: np.histogram(a, bins=vel_bins)[0],
                        0, inp_ds['doppler_velocity'].values[time_inds, 0:-2:2])).T
                
                if len(time_inds) > 1:
                    mask = np.squeeze(np.where(inp_ds[v].values[time_inds, :] > t, 1, 0))
                elif len(time_inds) == 0:
                    continue
                else:
                    mask = np.where(inp_ds[v].values[time_inds, :] > t, 1, 0)
                try:
                    mask = mask[:, 0:-2:2] + mask[:, 1:-1:2]
                    arr[i] = np.sum(mask, axis=0) / (2 * len(time_inds)) * 100.
                except:
                    continue
            scp_ds['%sgt%1.6f' % (v, t)] = xr.DataArray(arr, dims=('time', 'range'))
            scp_ds['%sgt%1.6f' % (v, t)].attrs["long_name"] = "Statistical coverage of %s > %f" % (v, t)
            scp_ds['snr_hist'] = xr.DataArray(hist_array, 
                    dims=('time', 'range', 'snr_bins'))
            scp_ds['vel_hist'] = xr.DataArray(hist_array_vel, 
                    dims=('time', 'range', 'vel_bins'))
    inp_ds.close()
    scp_ds['Time'] = xr.DataArray(time_range[:-1], dims=('time'))
    scp_ds['range_bins'] = xr.DataArray(range_bins[:-1], dims=('range'))
    scp_ds['snr_bins'] = xr.DataArray(snr_bins[:-1], dims=('snr_bins'))
    scp_ds['vel_bins'] = xr.DataArray(vel_bins[:-1], dims=('vel_bins'))
    scp_ds = xr.Dataset(scp_ds)
    print(scp_ds.range_bins)
    scp_ds.to_netcdf(out_path + 'sgpdlscp.%s.%s.nc' % (ymd, hms))
