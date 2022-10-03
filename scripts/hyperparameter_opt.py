import xarray as xr
import numpy as np
import pandas as pd

#from distributed import Client
import xgboost as xgb
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK
#from hyperopt.mongoexp import MongoTrials
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
        return 2
    elif pct_cloud < 0.5:
        return 0
    elif pct_cloud >= 0.5:
        return 1
    else:
        return 2
    
    raise ValueError("Invalid value %s for label" % my_string)

def dt64_to_dt(dt):
    ts = (dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)

def main():
    my_ds = xr.open_mfdataset('/lcrc/group/earthscience/rjackson/sgp_lidar/coverage_product/*.nc')
    labels = my_ds['time'].values
    labels = np.array([get_label(dt64_to_dt(x)) for x in labels])

    feature_list = ['snrgt3.000000', 'snrgt5.000000', 'snrgt10.000000']

    x = np.concatenate([my_ds[x].values[:, :150] for x in feature_list], axis=1)
    valid = np.where(labels > -1)[0]
    x = x[valid, :]
    labels = labels[valid]

    feature_labels = []
    for feat in feature_list:
        for i in range(len(my_ds.range_bins.values[:, :150])):
            feature_labels.append('%s%d' % (
                feat, i)) 

    print("Submitting jobs to queue...")
    
    
    print("Waiting for cluster...")
    #while client.ncores() == {}:
    #    continue
    #print(client)
    #print(my_cluster)
    print("Performing hyperparameter optimization...")

    #my_classifier = XGBClassifier(n_jobs=6, n_estimators=1000)
    #params = {'max_depth': 7,
    #          'colsample_bytree': 1,
    #          'subsample': 0.9,
    #          'gpu_id': 1,
    #          'eta': 0.01,
    #          'num_parallel_tree': 1,
    #          'tree_method': 'gpu_hist',
    #          'objective': 'multi:softmax',
    #          'num_class': 3,
    #          'verbosity': 1}

    params = {'eta': hp.uniform('eta', 0.01, 1),
              'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
              'num_rounds': hp.choice('num_rounds', 
                  np.arange(200, 2000, 100, dtype='int')),
              'gamma': hp.uniform('gamma', 0, 10),
              'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 1),
              'subsample': hp.uniform('subsample', 0.6, 1),
              }
              
    #dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_labels)
    #dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_labels)
    dall = xgb.DMatrix(x, label=labels)
    num_rounds = 10000

    def lr(boosting_round, num_boost_round):
        return max([0.0001, 0.1 - 0.1*boosting_round/num_boost_round])
    
    print("Training...")
    gpu_res = {}
    def objective(x):
        params = {'eta': x['eta'],
                  'max_depth': x['max_depth'],
                  'num_parallel_tree': 1,
                  'objective': 'multi:softmax',
                  'num_class': 3,
                  'tree_method': 'gpu_hist',
                  'colsample_bytree': x['colsample_bytree'],
                  'subsample': x['subsample'],
                  'gamma': x['gamma'],
                  'verbosity': 0}

        res = xgb.cv(params, dall, nfold=5,
                     num_boost_round=x['num_rounds'])
        return {'loss': res['test-merror-mean'].min(), 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective, space=params, algo=tpe.suggest, max_evals=1000,
             trials=trials) 
    print(best)

    #my_classifier.fit(x, labels)
    #print(scorer.cv_results_)
    #print(scorer.cv_results_['params'][scorer.best_index_])
    #results_df = pd.DataFrame(scorer.cv_results_)
    #results_dt.to_csv('hyperparameter_opt.csv')             
    
    #my_cluster.close()
    #del client
    
if __name__ == '__main__':
    main()

