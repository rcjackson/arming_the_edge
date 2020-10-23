import xarray as xr
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Add, Activation, Concatenate, Flatten
from tensorflow.keras.layers import Cropping2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from datetime import datetime

label_df = pd.read_csv('../notebooks/lidar_labels.csv')

date_list = np.array([datetime.strptime(x, '%Y-%m-%d').date() for x in label_df["Date"].values])
start_time_list = np.array([datetime.strptime(x[0:4], '%H%M').time() for x in label_df["Time"].values])
end_time_list = np.array([datetime.strptime(x[5:], '%H%M').time() for x in label_df["Time"].values])


def get_label(dt):
    label_ind = np.where(np.logical_and.reduce(
        (date_list == dt.date(), start_time_list <= dt.time(), end_time_list > dt.time())))
    if not label_ind[0].size:
        return -1
    my_strings = np.array(
            [x.lower() for x in label_df["Label"].values[label_ind]])
    num_cloud = len(np.where(my_strings == "cloudy")[0]) + \
            len(np.where(my_strings == "cloud")[0])
    num_clear = len(np.where(my_strings == "clear")[0])
    num_rain = len(np.where(my_strings == "rain")[0])
    my_string = label_df["Label"].values[label_ind][0]

    pct_cloud = num_cloud/len(my_strings)

    if num_rain > 0:
        return 3
    elif pct_cloud < 0.25:
        return 0
    elif pct_cloud >= 0.25 and pct_cloud <= 0.75:
        return 1
    else:
        return 2

    raise ValueError("Invalid value %s for label" % my_string)


def nn_classifier(input_array, num_layers=2, num_channel_start=2):
    inp = Input(shape=(input_array.shape[1],), name='input')
    layer = inp
    num_channels = num_channel_start
    for i in range(num_layers):
        layer = Dense(num_channels)(layer)
        layer = BatchNormalization()(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        #layer = Dropout(0.25)(layer)
        #layer = Dropout(0.75)(layer)
        #num_channels = num_channels * 2
        
    output = Dense(4, activation='softmax',
            name='output', kernel_initializer='he_normal')(layer)

    return Model(inp, output)


def dt64_to_dt(dt):
    ts = (dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)

def main():
    my_ds = xr.open_mfdataset('/lambda_stor/data/rjackson/coverage_product/*.nc')
    labels = my_ds['time'].values
    labels = np.array([get_label(dt64_to_dt(x)) for x in labels])
    print(my_ds.variables.keys()) 
    time = my_ds['time'].values
    range_bins = my_ds['range_bins'].values
    feature_list = ['snrgt3.000000', 'snrgt5.000000', 'snrgt10.000000']

    
    x = np.concatenate([my_ds[x].values for x in feature_list], axis=1)
    feature_labels = []
    for feat in feature_list:
        for i in range(len(my_ds.range_bins.values)):
            feature_labels.append('%s%d' % (feat, my_ds.range_bins[i]))

    valid = np.where(labels > -1)[0]
    x = x[valid, :]
    labels = labels[valid]
    time = time[valid]
    pct_clear = len(np.argwhere(labels == 0))/len(labels)
    # Since dataset is biased to clear values, remove half of clear values
    where_clear = np.argwhere(labels == 0)
    for inds in where_clear:
        if np.random.random(1) > pct_clear:
            labels[inds] = -1

    valid = np.where(labels > -1)[0]
    x = x[valid, :]
    labels = labels[valid]
    time = time[valid]

    x = np.where(np.isfinite(x), x, 0)
    x = scale(x)
    np.savez('snrscp.npz', x=x, labels=labels, time=time, range_bins=range_bins)
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.20)
    
    epoch_no = 1
    #y_train = tf.one_hot(y_train, depth=4).numpy()
    #y_test = tf.one_hot(y_test, depth=4).numpy()
    y = labels
    #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    model = nn_classifier(x_train, num_layers=12, num_channel_start=64)
    model.summary()
    model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(
        filepath=('/homes/rjackson/arming_the_edge/models/nnclassifier-{epoch:03d}.hdf5'),
        verbose=1)
    early_stop = EarlyStopping(patience=200)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3000, callbacks=[checkpointer, early_stop], initial_epoch=epoch_no)

    
    #history = pd.DataFrame(dicts['history'])
    

    #fname = 'train'
    #model = 'model_nonrf'
    #for feat in feature_list:
    #    fname += feat
    #    model += feat
    #fname += 'optimized_nonrf.csv'
    #model += '.json'
    #with open(fname, 'w') as f:
    #    f.write('Features used: ')
    #    for feat in feature_list:
    #        f.write('%s  ' % feat)
    #    f.write('\n')
    #    res.to_csv(f)
    
    #num_rounds = len(res['test-merror-mean'].values) 
    #bst = xgb.train(params, dtrain, num_boost_round=num_rounds, 
    #                callbacks=[xgb.callback.reset_learning_rate(lr)],
    #                evals=[(dtest, 'test')])

    #bst.save_model(model)
    #y_predict = bst.predict(dtest)
    #y_all_predict = bst.predict(dall)
    #y_train_predict = bst.predict(dtrain)
    y_all_predict = model.predict(x)
    y_predict = model.predict(x_test)
    y_train_predict = model.predict(x_train)

    predicted_labels_df = xr.Dataset({'label_true': labels, 'label_pred':
                                      y_all_predict,
                                      'label_train_pred': y_train_predict,
                                      'label_test_pred': y_predict,
                                      'label_train': y_train,
                                      'label_test': y_test})
    predicted_labels_df.to_netcdf('classification_nnn.nc')
    print("Accuracy score for test set: %f" % accuracy_score(y_test, y_predict))
    
    my_ds.close()

if __name__ == "__main__":
    main()
