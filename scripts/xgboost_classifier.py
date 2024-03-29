import xarray as xr
import pandas as pd
import numpy as np
import xgboost as xgb
import dask.array as da
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    my_string = label_df["Label"].values[label_ind][0]

    if my_string.lower() == 'clear':
        return 0
    elif my_string.lower() == 'cloudy' or my_string.lower() == "cloud":
        return 1
    elif my_string.lower() == 'rain':
        return 2

    raise ValueError("Invalid value %s for label" % my_string)

def dt64_to_dt(dt):
    ts = (dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)

def main():
    my_ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/sgp_lidar/coverage_product/*.nc')
    labels = my_ds['time'].values
    labels = np.array([get_label(dt64_to_dt(x)) for x in labels])
    
    
#{'colsample_bytree': 0.9591286458914207, 'eta': 0.26652131138562396, 'gamma': 2.6546371902167016, 'max_depth': 9, 'num_rounds': 2, 'subsample': 0.9255120113394381}

    params = {'max_depth': 11,
              'colsample_bytree': 0.95128645814207,
              'subsample': 0.9255120113394381,
              'eta': 0.26652131138562396,
              'gamma': 2.6546371902167016,
              'num_parallel_tree': 1,
              'tree_method': 'gpu_hist',
              'objective': 'multi:softmax',
              'num_class': 3,
              'verbosity': 1}
    feature_list = []
    for c in sys.argv[1:]:
        feature_list.append('snrgt%s.000000' % c)
    print(feature_list)
    print(my_ds)
    x = np.concatenate([my_ds[x].values[:, :150] for x in feature_list], axis=1)
    feature_labels = []
    for feat in feature_list:
        for i in range(len(my_ds.range_bins[0,:150].values)):
            feature_labels.append('%s%d' % (feat, my_ds.range_bins[0,i]))

    valid = np.where(labels > -1)[0]
    x = x[valid, :]
    labels = labels[valid]
    print('%d valid points' % len(labels))

    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.20)
    print(x_train.shape, y_train.shape)

    #x_train = da.from_array(x_train, chunks=(1000, len(feature_labels)))
    #x_test = da.from_array(x_test, chunks=(1000, len(feature_labels)))
    #y_train = da.from_array(y_train, chunks=(1000,))
    #y_test = da.from_array(y_test, chunks=(1000,))
    
    #x = da.from_array(x, chunks=(1000,len(feature_labels)))
    #labels = da.from_array(labels, chunks=(1000,))
    print("Loading arrays into dask cluster...")
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_labels)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_labels)
    dall = xgb.DMatrix(x, label=labels, feature_names=feature_labels)
    num_rounds = 1500

    def lr(boosting_round, num_boost_round):
        return max([0.0001, 0.1 - 0.1*boosting_round/num_boost_round])

    print("Training...")
    gpu_res = {}
    res = xgb.cv(params, dall, nfold=5,
                 num_boost_round=num_rounds,
                 callbacks=[xgb.callback.print_evaluation(1), 
                            xgb.callback.early_stop(50),
                            ])
    
    #history = pd.DataFrame(dicts['history'])
    

    fname = 'train'
    model = 'model_nonrf'
    for feat in feature_list:
        fname += feat
        model += feat
    feats = fname
    fname += 'optimized_nonrf.csv'
    model += '.json'
    with open(fname, 'w') as f:
        f.write('Features used: ')
        for feat in feature_list:
            f.write('%s  ' % feat)
        f.write('\n')
        res.to_csv(f)
    
    num_rounds = len(res['test-merror-mean'].values) 
    bst = xgb.train(params, dtrain, num_boost_round=num_rounds, 
                    callbacks=[xgb.callback.reset_learning_rate(lr)],
                    evals=[(dtest, 'test')])

    bst.save_model(model)
    y_predict = bst.predict(dtest)
    y_all_predict = bst.predict(dall)
    y_train_predict = bst.predict(dtrain)
    predicted_labels_df = xr.Dataset({'label_true': labels, 'label_pred':
                                      y_all_predict,
                                      'label_train_pred': y_train_predict,
                                      'label_test_pred': y_predict,
                                      'label_train': y_train,
                                      'label_test': y_test})
    
    predicted_labels_df.to_netcdf("classifications/classification_%s.nc" % feats)
    print("Accuracy score for test set: %f" % accuracy_score(y_test, y_predict))
    
    my_ds.close()

if __name__ == "__main__":
    main()
