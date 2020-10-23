import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
X = np.load('./snrscp.npz')
X.files
m = load_model('../models/nnclassifier-055.hdf5')
x = X['x']
valid_x = np.all(np.isfinite(x), axis=1)
x = x[valid_x]
y = m.predict(x)
y = np.argmax(y, axis=1)
print(y)
