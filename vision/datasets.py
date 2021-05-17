from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

N_MODALITIES = 6
VALID_PARTITIONS = {'train': 0, 'val': 1, 'test': 2}


class TCGADataset(Dataset):
    """TCGA Landmarks dataset."""

    def __init__(self, partition='train'):
        """
        Args:
            partition (string): Determines if this dataset is used for training or testing
        """
        self.split = partition
        print("Getting {} Dataset".format(self.split))
        split_per = 0.1

        rna_file = "/Users/bram/rp-group-21-bpronk/data/TCGA/RNA_COMMONGNC.csv"
        rna_data = pd.read_csv(rna_file, usecols=range(1, 5001))
        print("-----   RNA file read   -----")
        print("RNA SHAPE : ", rna_data.shape)

        rna_test_split = rna_data.sample(frac=split_per)

        self.rna_test_file = np.nan_to_num(np.float32(rna_test_split.to_numpy()))
        self.rna_train_file = np.nan_to_num(np.float32(rna_data.to_numpy()))

        gcn_file = "/Users/bram/rp-group-21-bpronk/data/TCGA/GNC_POSITIVE_COMMONRNA.csv"
        gcn_data = pd.read_csv(gcn_file, usecols=range(1, 5001))
        print("-----   GCN file read   -----")
        print("GCN SHAPE : ", gcn_data.shape)

        gcn_test_split = gcn_data.sample(frac=split_per)

        self.gcn_test_file = np.nan_to_num(np.float32(gcn_test_split.to_numpy()))
        self.gcn_train_file = np.nan_to_num(np.float32(gcn_data.to_numpy()))

    def __len__(self):
        if self.split == 'train':
            return self.rna_train_file.shape[0]
        else:
            return self.rna_test_file.shape[0]

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.rna_train_file[idx], self.gcn_train_file[idx]
        else:
            return self.rna_test_file[idx], self.gcn_test_file[idx]
