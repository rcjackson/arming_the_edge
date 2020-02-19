import xarray as xr
import numpy as np
import pandas as pd

from dask_jobqueue import SLURMCluster
from distributed import Client
from xgboost import XGBClassifier
from dask_ml.model_selection import GridSearchCV


def main():
    my_ds = xr.open_dataset('../notebooks/encoded_snr_parameters.nc')
    encoded_data_test = my_ds.encoded_data_test.values
    encoded_data_train = my_ds.encoded_data_train.values
    test_labels = my_ds.test_labels
    train_labels = my_ds.train_labels

    nonans_test = ~np.isnan(test_labels)
    nonans_train = ~np.isnan(train_labels)
    encoded_data_test = encoded_data_test[nonans_test, :]
    encoded_data_train = encoded_data_train[nonans_train, :]
    test_labels = test_labels[nonans_test]
    train_labels = train_labels[nonans_train]

    print("Submitting jobs to queue...")
    my_cluster = SLURMCluster(memory='64GB', cores=36, walltime='4:00:00', processes=6)
    my_cluster.scale(32)
    client = Client(my_cluster)
    
    print("Waiting for cluster...")
    while client.ncores() == {}:
        continue
    print(client)
    print(my_cluster)
    print("Performing hyperparameter optimization...")

    my_classifier = XGBClassifier(n_jobs=6)
    params = {'learning_rate': np.arange(0.01, 0.2, 0.1),
              'max_depth': np.arange(1, 20, 1, dtype='int'),
              'n_estimators': np.arange(10, 1000, 50, dtype='int')}
    scorer = GridSearchCV(estimator=my_classifier, param_grid=params)
    scorer.fit(encoded_data_train, train_labels)    

    print(scorer.cv_results_)
    print(scorer.cv_results_['params'][scorer.best_index_])
    results_df = pd.DataFrame(scorer.cv_results_)
    results_dt.to_csv('hyperparameter_opt.csv')             
    
    my_cluster.close()
    del client
    
if __name__ == '__main__':
    main()

