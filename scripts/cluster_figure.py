from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Gen = ImageDataGenerator(rescale=1/255.)
dataset = Gen.flow_from_directory(
    '/lcrc/group/earthscience/rjackson/lidar_pngs/5min_snr/training', class_mode='input', target_size=(256, 192), shuffle=False)


#inp, encoder, decoded = encoder_decoder_model()
model = load_model('/lcrc/group/earthscience/rjackson/arming_the_edge/encoder/encoder-decoder-1999.hdf5')
model.summary()
encoder = Model(model.get_layer("input").output, model.get_layer("encoding").output)
encodings = encoder.predict(dataset)

encodings = encodings.reshape((encodings.shape[0], np.prod(encodings.shape[1:])))
encodings.shape
model = PCA(n_components=10)
model.fit(encodings)
reduced = model.transform(encodings)
reduced.shape

inertia = np.zeros(39)
for i in range(1, 40):
    km = KMeans(n_clusters=i)
    km.fit(reduced)
    inertia[i - 1] = km.inertia_

plt.figure(figsize=(7, 4))
plt.plot(range(1, 40), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.ylim([0, 500000])
plt.savefig('Clusters.png', dpi=300, bbox_inches='tight')

