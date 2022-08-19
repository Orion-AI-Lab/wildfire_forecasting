import numpy.ma as ma
import torch
import csv
import warnings
import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import fsspec
import zarr
import os
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
import random
import json
import random

target = 'burned_areas'


class FireDataset_npy(Dataset):
    def __init__(self, dataset_root: str = None, access_mode: str = 'spatiotemporal',
                 problem_class: str = 'classification',
                 train_val_test: str = 'train', dynamic_features: list = None, static_features: list = None,
                 categorical_features: list = None, nan_fill: float = -1., neg_pos_ratio: int = 2, clc: str = None):
        """
        @param dataset_root: str where the dataset resides. It must contain also the minmax_clc.json
                and the variable_dict.json
        @param access_mode: spatial, temporal or spatiotemporal
        @param problem_class: classification or segmentation
        @param train_val_test:
                'train' gets samples from [2009-2018].
                'val' gets samples from 2019.
                test' get samples from 2020
        @param dynamic_features: selects the dynamic features to return
        @param static_features: selects the static features to return
        @param categorical_features: selects the categorical features
        @param nan_fill: Fills nan with the value specified here
        """
        # dataset_root should be a str leading to the path where the data have been downloaded and decompressed
        # Make sure to follow the details in the readme for that
        if not dataset_root:
            raise ValueError('dataset_root variable must be set. Check README')
        dataset_root = Path(dataset_root)
        min_max_file = dataset_root / 'minmax_clc.json'
        variable_file = dataset_root / 'variable_dict.json'

        with open(min_max_file) as f:
            self.min_max_dict = json.load(f)

        with open(variable_file) as f:
            self.variable_dict = json.load(f)

        if static_features is None:
            static_features = all_static_features
        if dynamic_features is None:
            dynamic_features = all_dynamic_features
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.categorical_features = categorical_features
        self.access_mode = access_mode
        self.problem_class = problem_class
        self.nan_fill = nan_fill
        self.clc = clc
        assert problem_class in ['classification', 'segmentation']
        if problem_class == 'classification':
            self.target = 'burned'
        else:
            self.target = 'burned_areas'
        assert self.access_mode in ['spatial', 'temporal', 'spatiotemporal']
        dataset_path = dataset_root / 'npy' / self.access_mode
        self.positives_list = list((dataset_path / 'positives').glob('*dynamic.npy'))
        self.positives_list = list(zip(self.positives_list, [1] * (len(self.positives_list))))
        val_year = 2020
        test_year = min(val_year + 1, 2021)

        self.train_positive_list = [(x, y) for (x, y) in self.positives_list if int(x.stem[:4]) < val_year]
        self.val_positive_list = [(x, y) for (x, y) in self.positives_list if int(x.stem[:4]) == val_year]
        self.test_positive_list = [(x, y) for (x, y) in self.positives_list if int(x.stem[:4]) == test_year]

        self.negatives_list = list((dataset_path / 'negatives_clc').glob('*dynamic.npy'))
        self.negatives_list = list(zip(self.negatives_list, [0] * (len(self.negatives_list))))

        self.train_negative_list = random.sample(
            [(x, y) for (x, y) in self.negatives_list if int(x.stem[:4]) < val_year],
            len(self.train_positive_list) * neg_pos_ratio)
        self.val_negative_list = random.sample(
            [(x, y) for (x, y) in self.negatives_list if int(x.stem[:4]) == val_year],
            len(self.val_positive_list) * neg_pos_ratio)

        self.negatives_list = list((dataset_path / 'negatives_clc').glob('*dynamic.npy'))
        self.negatives_list = list(zip(self.negatives_list, [0] * (len(self.negatives_list))))
        self.test_negative_list = random.sample(
            [(x, y) for (x, y) in self.negatives_list if int(x.stem[:4]) == test_year],
            len(self.test_positive_list) * neg_pos_ratio)

        self.dynamic_idxfeat = [(i, feat) for i, feat in enumerate(self.variable_dict['dynamic']) if
                                feat in self.dynamic_features]
        self.static_idxfeat = [(i, feat) for i, feat in enumerate(self.variable_dict['static']) if
                               feat in self.static_features]
        self.dynamic_idx = [x for (x, _) in self.dynamic_idxfeat]
        self.static_idx = [x for (x, _) in self.static_idxfeat]

        if train_val_test == 'train':
            print(f'Positives: {len(self.train_positive_list)} / Negatives: {len(self.train_negative_list)}')
            self.path_list = self.train_positive_list + self.train_negative_list
        elif train_val_test == 'val':
            print(f'Positives: {len(self.val_positive_list)} / Negatives: {len(self.val_negative_list)}')

            self.path_list = self.val_positive_list + self.val_negative_list
        elif train_val_test == 'test':
            print(f'Positives: {len(self.test_positive_list)} / Negatives: {len(self.test_negative_list)}')

            self.path_list = self.test_positive_list + self.test_negative_list
        print("Dataset length", len(self.path_list))
        random.shuffle(self.path_list)
        self.mm_dict = self._min_max_vec()

    def _min_max_vec(self):
        mm_dict = {'min': {}, 'max': {}}
        for agg in ['min', 'max']:
            if self.access_mode == 'spatial':
                mm_dict[agg]['dynamic'] = np.ones((len(self.dynamic_features), 1, 1))
                mm_dict[agg]['static'] = np.ones((len(self.static_features), 1, 1))
                for i, (_, feat) in enumerate(self.dynamic_idxfeat):
                    mm_dict[agg]['dynamic'][i, :, :] = self.min_max_dict[agg][self.access_mode][feat]
                for i, (_, feat) in enumerate(self.static_idxfeat):
                    mm_dict[agg]['static'][i, :, :] = self.min_max_dict[agg][self.access_mode][feat]

            if self.access_mode == 'temporal':
                mm_dict[agg]['dynamic'] = np.ones((1, len(self.dynamic_features)))
                mm_dict[agg]['static'] = np.ones((len(self.static_features)))
                for i, (_, feat) in enumerate(self.dynamic_idxfeat):
                    mm_dict[agg]['dynamic'][:, i] = self.min_max_dict[agg][self.access_mode][feat]
                for i, (_, feat) in enumerate(self.static_idxfeat):
                    mm_dict[agg]['static'][i] = self.min_max_dict[agg][self.access_mode][feat]

            if self.access_mode == 'spatiotemporal':
                mm_dict[agg]['dynamic'] = np.ones((1, len(self.dynamic_features), 1, 1))
                mm_dict[agg]['static'] = np.ones((len(self.static_features), 1, 1))
                for i, (_, feat) in enumerate(self.dynamic_idxfeat):
                    mm_dict[agg]['dynamic'][:, i, :, :] = self.min_max_dict[agg][self.access_mode][feat]
                for i, (_, feat) in enumerate(self.static_idxfeat):
                    mm_dict[agg]['static'][i, :, :] = self.min_max_dict[agg][self.access_mode][feat]
        return mm_dict

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path, labels = self.path_list[idx]
        dynamic = np.load(path)
        static = np.load(str(path).replace('dynamic', 'static'))
        if self.access_mode == 'spatial':
            dynamic = dynamic[self.dynamic_idx]
            static = static[self.static_idx]
        elif self.access_mode == 'temporal':
            dynamic = dynamic[:, self.dynamic_idx, ...]
            static = static[self.static_idx]
        else:
            dynamic = dynamic[:, self.dynamic_idx, ...]
            static = static[self.static_idx]

        def _min_max_scaling(in_vec, max_vec, min_vec):
            return (in_vec - min_vec) / (max_vec - min_vec)

        dynamic = _min_max_scaling(dynamic, self.mm_dict['max']['dynamic'], self.mm_dict['min']['dynamic'])
        static = _min_max_scaling(static, self.mm_dict['max']['static'], self.mm_dict['min']['static'])

        if self.access_mode == 'temporal':
            feat_mean = np.nanmean(dynamic, axis=0)
            # Find indices that you need to replace
            inds = np.where(np.isnan(dynamic))
            # Place column means in the indices. Align the arrays using take
            dynamic[inds] = np.take(feat_mean, inds[1])

        elif self.access_mode == 'spatiotemporal':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                feat_mean = np.nanmean(dynamic, axis=(2, 3))
                feat_mean = feat_mean[..., np.newaxis, np.newaxis]
                feat_mean = np.repeat(feat_mean, dynamic.shape[2], axis=2)
                feat_mean = np.repeat(feat_mean, dynamic.shape[3], axis=3)
                dynamic = np.where(np.isnan(dynamic), feat_mean, dynamic)
        if self.nan_fill:
            dynamic = np.nan_to_num(dynamic, nan=self.nan_fill)
            static = np.nan_to_num(static, nan=self.nan_fill)

        if self.clc == 'mode':
            clc = np.load(str(path).replace('dynamic', 'clc_mode'))
        elif self.clc == 'vec':
            clc = np.load(str(path).replace('dynamic', 'clc_vec'))
            clc = np.nan_to_num(clc, nan=0)
        else:
            clc = 0
        return dynamic, static, clc, labels
