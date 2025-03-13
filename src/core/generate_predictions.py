import glob

import geopandas as gpd
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from libpysal.graph import read_parquet
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from core.utils import used_keys

from palettable.colorbrewer.qualitative import Set3_12
from sklearn.metrics import davies_bouldin_score, f1_score

from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn import model_selection
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer

coredir = '/data/uscuni-eurofab-overture/'

def get_level_cut(mapping_level, v = 'v3'):
    
    cluster_mapping = pd.read_parquet(f'/data/uscuni-ulce/processed_data/clusters/cluster_mapping_{v}.pq')

    if mapping_level == 3:
        level_cut = cluster_mapping[3].astype(str)
        level_cut[level_cut == '2'] = '8'
    
    elif mapping_level == 4:
        # # assign outliers to the industrial cluster
        level_cut = cluster_mapping[4].astype(str)
        level_cut[level_cut == '3'] = '15'
        level_cut[level_cut == '4'] = '15'
        level_cut[level_cut == '10'] = '15'
        
    return level_cut


def read_train_test(train_test_iteration, mapping_level, sample_size):
    
    X_train = pd.read_parquet(f'{coredir}processed_data/train_test_data/training_data{train_test_iteration}.pq')
    y = pd.read_parquet(f'{coredir}processed_data/train_test_data/training_labels{train_test_iteration}.pq')
    level_cut = get_level_cut(mapping_level)

    y['final_without_noise'] = y['final_without_noise'].map(level_cut.to_dict())

    ### undersample
    if sample_size > y['final_without_noise'].value_counts().iloc[-1]:
        sample_size = y['final_without_noise'].value_counts().iloc[-1] - 1_000
    
    np.random.seed(123)
    train_indices = []
    classes = y.final_without_noise.unique()
    has_building = ~y.index.str.split('_').str[-1].str.startswith('-')
    
    for cluster in classes:
        random_indices = np.random.choice(np.where((y.final_without_noise == cluster) & (has_building))[0], sample_size, replace=False, )
        train_indices.append(random_indices)
    
    train_indices = np.concat(train_indices)
    X_train = X_train.iloc[train_indices]
    y = y.iloc[train_indices]
    assert y.final_without_noise.isna().sum() == 0

    X_resampled, y_resampled = X_train, y.final_without_noise
    if 'source' in X_resampled.columns:
        # we can do this because of random forest splitting
        source_factorizer = X_resampled['source'].factorize()
        X_resampled['source'] = source_factorizer[0]
    print(y_resampled.value_counts())

    return X_resampled, y_resampled


def get_cluster_names(mapping_level):
    
    if mapping_level == 3:
        cluster_names = {
        '1': 'Central Urban Developments',
        '2': 'Large Scale Outliers',
         '3': 'Dense Urban Developments',
         '4': 'Street-aligned Developments',
         '5': 'Sparse Rural Development',
         '6': 'Linear Road Network Developments',
         '7': 'Sparse Road Network Developments',
         '8': 'Large Scale Developments'
        }

    elif mapping_level == 4:
        # # assign outliers to the industrial cluster
        cluster_names = {'1': 'Dense Connected Developments',
         '2': 'Large Interconnected Blocks',
         '3': 'Extensive Courtyard Complexes',
         '4': 'Massive Connected Aggregations',
         '5': 'Dense Standalone Buildings',
         '6': 'Compact Development',
         '7': 'Cul-de-Sac Layout',
         '8': 'Aligned Winding Streets',
         '9': 'Sparse Rural Development',
         '10': 'Large Wide-Spaced Complexes',
         '11': 'Dispersed Linear Development',
         '12': 'Linear Development',
         '13': 'Sparse Open Layout',
         '14': 'Sparse Road-Linked Development',
         '15': 'Large Utilitarian Development',
         '16': 'Extensive Wide-Spaced Developments'}

    return cluster_names


def score_predictions(train_test_iteration, mapping_level, model):
    
    level_cut = get_level_cut(mapping_level)
    X_test = pd.read_parquet(f'{coredir}processed_data/train_test_data/testing_data{train_test_iteration}.pq')
    y_test = pd.read_parquet(f'{coredir}processed_data/train_test_data/testing_labels{train_test_iteration}.pq')
    y_test['final_without_noise'] = y_test['final_without_noise'].map(level_cut.to_dict())

    cluster_names = get_cluster_names(mapping_level)
    
    assert y_test.final_without_noise.isna().sum() == 0
    assert (X_test.index == y_test.index).all()
    
    print(y_test.final_without_noise.map(cluster_names).value_counts())

    if 'source' in X_test.columns:
        # we can do this because of random forest splitting
        factorizer_dict = pd.Series(np.arange(len(source_factorizer[1])), source_factorizer[1].values, ).to_dict()
        X_test['source'] = X_test['source'].map(factorizer_dict)

    ## predictions
    predictions = model.predict(X_test)

    weighted = f1_score(y_test, predictions, average='weighted')
    micro = f1_score(y_test, predictions, average='micro')
    macro = f1_score(y_test, predictions, average='macro')
    overall_acc = pd.Series([weighted, micro, macro], index=['Weighted F1', 'Micro F1', 'Macro F1'])

    f1s_vals = f1_score(y_test, predictions, average=None)
    f1s = pd.Series(
        f1s_vals,
        index = [cluster_names[k] for k in sorted(np.unique(predictions))]
    )
    f1s = f1s.sort_values()

    print(overall_acc)
    print(f1s)
    overall_acc.to_csv(f'{coredir}processed_data/results/overall_acc_{mapping_level}_{train_test_iteration}.csv')
    f1s.to_csv(f'{coredir}processed_data/results/class_f1s_{mapping_level}_{train_test_iteration}.csv')
    

def train_model(train_test_iteration, mapping_level, sample_size):

    X_resampled, y_resampled = read_train_test(train_test_iteration, mapping_level, sample_size)

    
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    model = HistGradientBoostingClassifier(random_state=123, verbose=1,
                                                learning_rate = 0.03,
                                                max_depth = None, 
                                                max_iter = 120, 
                                                max_leaf_nodes=None,
                                                max_features=.5
                                               )
    model.fit(X_resampled, y_resampled)
    print(model.score(X_resampled, y_resampled))


    
    score_predictions(train_test_iteration, mapping_level, model)


if __name__ == '__main__':

    mapping_level = 3
    sample_size = 600_000
    
    # for train_test_iteration in range(1, 8):
    #     train_model(train_test_iteration, mapping_level, sample_size)

    train_model(7, 3, sample_size)
    train_model(6, 3, sample_size)

    train_model(7, 4, sample_size)
    train_model(6, 4, sample_size)
