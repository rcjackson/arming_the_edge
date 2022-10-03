import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ceil_path = '/lcrc/group/earthscience/rjackson/sgp_pblh/*.nc'
classifications = '/lcrc/group/earthscience/rjackson/arming_the_edge/notebooks/Clusters_new10.nc'

clusters = xr.open_dataset(classifications)

# Convert to local time (CDT = UTC - 5)
times = clusters.time.dt.hour - 5
times[times < 0] = times[times < 0] + 24
print(times)
bins = np.arange(-0.5, 24.5, 1)
#cbh_hist0, cbh_bins0 = np.histogram(cbh.values[where0], bins, normed=True)
#cbh_hist1, cbh_bins1 = np.histogram(cbh.values[where1], bins, normed=True)
#cbh_hist2, cbh_bins2 = np.histogram(cbh.values[where2], bins, normed=True)
#cbh_hist3, cbh_bins3 = np.histogram(cbh.values[where3], bins, normed=True)
#cbh_hist4, cbh_bins4 = np.histogram(cbh.values[where4], bins, normed=True)
#cbh_hist5, cbh_bins5 = np.histogram(cbh.values[where5], bins, normed=True)
clusters.close()

fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    where = np.argwhere(clusters.classification.values == i)
    cbh_hist0, cbh_bins0 = np.histogram(times.values[where], bins, normed=False)
    ax[int(i/5), i % 5].plot(cbh_bins0[:-1], cbh_hist0,
            label='Cluster %d' % i, linewidth=2)
    if int(i/5) == 1:
        ax[int(i/5), i % 5].set_xticks([0, 6, 12, 18])
        ax[int(i/5), i % 5].set_xlabel('Hour [local time]')
    else:
        ax[int(i/5), i % 5].set_xticks([])
    if i % 5 > 0:
        ax[int(i/5), i % 5].set_yticks([])
    else:
        ax[int(i/5), i % 5].set_ylabel('Count')
    ax[int(i/5), i % 5].set_xlim([0, 24])
    ax[int(i/5), i % 5].set_ylim([0, 600])
    ax[int(i/5), i % 5].set_title(str(i + 1))
#plt.plot(cbh_bins5[:-1], cbh_hist5, label='Cluster 6', linewidth=2)
fig.savefig('time_hist.png', bbox_inches='tight')
