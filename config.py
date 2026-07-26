import os

import torch

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SGCC_RAW_PATH = os.path.join(DATA_DIR, 'raw_data.csv')
OEDI_RAW_PATH = os.path.join(DATA_DIR, 'df.csv')

SGCC_CONFIG = {
    'name': 'sgcc',
    'n_users': 42372,
    'n_days': 1035,
    'theft_rate': 0.0853,
    'missing_rate': 0.256,
    'label_col': 'FLAG',
    'id_col': 'CONS_NO',
    'channels': 5,
    'seq_len': 1035,
    'gbdt_params': {
        'lgb': {
            'n_estimators': 2000,
            'max_depth': 7,
            'learning_rate': 0.03,
            'num_leaves': 63,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'min_child_samples': 50,
            'scale_pos_weight': 10.72,
            'random_state': 42,
            'verbose': -1,
        },
        'xgb': {
            'n_estimators': 1500,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'min_child_weight': 5,
            'scale_pos_weight': 10.72,
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0,
        },
        'catboost': {
            'iterations': 1500,
            'depth': 7,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3.0,
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': 0,
        },
    },
    'tcn_params': {
        'num_channels': [32, 32, 32, 16],
        'kernel_size': 5,
        'dropout': 0.2,
        'leaf_embed_dim': 4,
        'n_trees': 100,
        'num_leaves': 64,
    },
    'train_params': {
        'batch_size': 64,
        'epochs': 50,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'patience': 10,
        'focal_alpha': 0.85,
        'focal_gamma': 2.0,
        'recall_weight': 3.0,
    },
}

OEDI_CONFIG = {
    'name': 'oedi',
    'window_hours': 720,
    'step_hours': 168,
    'n_features': 10,
    'label_col': 'theft',
    'id_col': 'Class',
    'channels': 11,
    'theft_rate': 0.4081,
    'theft_types': ['Theft1', 'Theft2', 'Theft3', 'Theft4', 'Theft5', 'Theft6'],
    'building_types': [
        'FullServiceRestaurant', 'Hospital', 'LargeHotel', 'LargeOffice',
        'MediumOffice', 'OutPatient', 'PrimarySchool', 'QuickServiceRestaurant',
        'RetailStore', 'SecondarySchool', 'SmallHotel', 'SmallOffice',
        'SuperMarket', 'Warehouse',
    ],
    'gbdt_params': {
        'lgb': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': SEED,
            'verbose': -1,
        },
        'xgb': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': SEED,
            'tree_method': 'hist',
            'verbosity': 0,
        },
        'catboost': {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.03,
            'random_seed': SEED,
            'verbose': 0,
        },
    },
    'tcn_params': {
        'num_channels': [64, 64, 64, 32],
        'kernel_size': 5,
        'dropout': 0.2,
        'leaf_embed_dim': 64,
        'n_trees': 100,
        'num_leaves': 31,
    },
    'train_params': {
        'batch_size': 64,
        'epochs': 40,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'patience': 5,
        'focal_alpha': 0.55,
        'focal_gamma': 1.0,
        'recall_weight': 1.5,
    },
}