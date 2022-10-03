import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ceil_path = '/lcrc/group/earthscience/rjackson/sgpceilC1.b1/*.nc'
classifications = '/lcrc/group/earthscience/rjackson/arming_the_edge/notebooks/Clusters_new10.nc'

clusters = xr.open_dataset(classifications)
ceil_ds = xr.open_mfdataset(ceil_path)
print(clusters.classification)
cbh = ceil_ds['first_cbh'].load()
clusters = clusters.reindex(time=ceil_ds.time, method='nearest',
        tolerance=300e9)

print(clusters.classification)
print(np.sum(clusters.classification.values == 1))
#where0 = np.argwhere(clusters.classification.values == 0)
#where1 = np.argwhere(clusters.classification.values == 1)
#where2 = np.argwhere(clusters.classification.values == 2)
#where3 = np.argwhere(clusters.classification.values == 3)
#where4 = np.argwhere(clusters.classification.values == 4)
#where5 = np.argwhere(clusters.classification.values == 5)

bins = np.linspace(0, 8000., 30)
#cbh_hist0, cbh_bins0 = np.histogram(cbh.values[where0], bins, normed=True)
#cbh_hist1, cbh_bins1 = np.histogram(cbh.values[where1], bins, normed=True)
#cbh_hist2, cbh_bins2 = np.histogram(cbh.values[where2], bins, normed=True)
#cbh_hist3, cbh_bins3 = np.histogram(cbh.values[where3], bins, normed=True)
#cbh_hist4, cbh_bins4 = np.histogram(cbh.values[where4], bins, normed=True)
#cbh_hist5, cbh_bins5 = np.histogram(cbh.values[where5], bins, normed=True)
clusters.close()
ceil_ds.close()

fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    where = np.argwhere(clusters.classification.values == i)
    cbh_hist0, cbh_bins0 = np.histogram(cbh.values[where], bins, normed=False)
    pct_nan = len(np.argwhere(~np.isfinite(cbh.values[where]))) / len(where) * 100
    ax[int(i/5), i % 5].plot(cbh_hist0, cbh_bins0[:-1], label='Cluster %d' % i, linewidth=2)
    if i % 5 == 0:
        ax[int(i/5), i % 5].set_yticks([0, 2000, 4000, 6000, 8000])
        ax[int(i/5), i % 5].set_ylabel('Cloud base height [m]')
    else:
        ax[int(i/5), i % 5].set_yticks([])
    if i < 5:
        ax[int(i/5), i % 5].set_xticks([])
    else:
        ax[int(i/5), i % 5].set_xlabel('Count')
    #ax[int(i/6)].legend()
    ax[int(i/5), i % 5].set_xlim([0, 6000])
    ax[int(i/5), i % 5].text(4000, 7000, '%3.2f' % pct_nan)
    ax[int(i/5), i % 5].set_title(str(i + 1))
#plt.plot(cbh_bins4[:-1], cbh_hist0, label='Cluster 1', linewidth=2)
#plt.plot(cbh_bins5[:-1], cbh_hist1, label='Cluster 2', linewidth=2)
#plt.plot(cbh_bins4[:-1], cbh_hist2, label='Cluster 3', linewidth=2)
#plt.plot(cbh_bins5[:-1], cbh_hist3, label='Cluster 4', linewidth=2)
#plt.plot(cbh_bins4[:-1], cbh_hist4, label='Cluster 5', linewidth=2)
#plt.plot(cbh_bins5[:-1], cbh_hist5, label='Cluster 6', linewidth=2)
fig.savefig('cbh_hist.png', bbox_inches='tight')
