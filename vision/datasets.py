from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

TRAINING_DATA_SPLIT = 0.7
VALIDATION_DATA_SPLIT = 0.1
PREDICT_DATA_SPLIT = 0.2
VALID_PARTITIONS = {'train': 0, 'val': 1}


class TCGAData(object):
    """TCGA Landmarks dataset."""

    def __init__(self, save_dir=None, indices_path=None):
        """
        Args:
            save_dir     (string) : Where the indices taken from the datasets should be saved
            indices_path (string) : If set, use predefined indices for data split
        """
        # Datasets are assumed to be pre-processed and have the same ordering of samples

        # RNA-seq
        rna_file = "/Users/bram/Desktop/CSE3000/data/common/RNASeqV2_3000MAD_COMMON.csv"
        rna_data = pd.read_csv(rna_file, usecols=range(1, 3001))
        print("-----   RNA file read   -----")

        # Gene Copy Number
        gcn_file = "/Users/bram/Desktop/CSE3000/data/common/GCN_Gistic2_3000MAD_COMMON.csv"
        gcn_data = pd.read_csv(gcn_file, usecols=range(1, 3001))
        print("-----   GCN file read   -----")

        # DNA Methylation
        dna_file = "/Users/bram/Desktop/CSE3000/data/common/DNAMethylation_3000MAD_COMMON.csv"
        dna_data = pd.read_csv(dna_file, usecols=range(1, 3001))
        print("-----   DNA file read   -----")

        # Split Datasets into a 70 training / 10 validation / 20 prediction split
        assert rna_data.shape[0] == gcn_data.shape[0] == dna_data.shape[0], "Datasets do not have equal samples"

        # Split the dataset
        if indices_path is None:
            nr_of_samples = rna_data.shape[0]
            nr_of_training_samples = int(TRAINING_DATA_SPLIT * nr_of_samples)
            nr_of_validation_samples = int(VALIDATION_DATA_SPLIT * nr_of_samples)

            # Random ordering of all sample id's
            random_sample_indices = np.random.choice(a=nr_of_samples, size=nr_of_samples, replace=False)

            # Split into three sets of sizes
            # [:nr_of_training_samples], [nr_of_training_samples:nr_of_validation_samples], [:nr_of_predict_samples]
            sets = np.split(random_sample_indices,
                            [nr_of_training_samples, (nr_of_training_samples + nr_of_validation_samples)])

            training_ids = sets[0]
            validation_ids = sets[1]
            predict_ids = sets[2]

            if save_dir is None:
                print("Error, no save path is given so indices for data splits could not be saved. Exiting program")
                sys.exit()

            # Save the indices taken for reproducibility
            np.save("{}/training_indices.npy".format(save_dir), training_ids)
            np.save("{}/validation_indices.npy".format(save_dir), validation_ids)
            np.save("{}/predict_indices.npy".format(save_dir), predict_ids)

        else:  # Use predefined indices
            print("Using Predefined split")
            training_ids = np.load("{}/training_indices.npy".format(indices_path))
            validation_ids = np.load("{}/validation_indices.npy".format(indices_path))
            predict_ids = np.load("{}/predict_indices.npy".format(indices_path))

        # Create data arrays
        self.rna_train_file = np.nan_to_num(np.float32(rna_data.iloc[training_ids].to_numpy()))
        self.rna_val_file = np.nan_to_num(np.float32(rna_data.iloc[validation_ids].to_numpy()))
        self.rna_predict_file = np.nan_to_num(np.float32(rna_data.iloc[predict_ids].to_numpy()))

        self.gcn_train_file = np.nan_to_num(np.float32(gcn_data.iloc[training_ids].to_numpy()))
        self.gcn_val_file = np.nan_to_num(np.float32(gcn_data.iloc[validation_ids].to_numpy()))
        self.gcn_predict_file = np.nan_to_num(np.float32(gcn_data.iloc[predict_ids].to_numpy()))

        self.dna_train_file = np.nan_to_num(np.float32(dna_data.iloc[training_ids].to_numpy()))
        self.dna_val_file = np.nan_to_num(np.float32(dna_data.iloc[validation_ids].to_numpy()))
        self.dna_predict_file = np.nan_to_num(np.float32(dna_data.iloc[predict_ids].to_numpy()))

    def get_data_partition(self, partition):
        if partition == "train":
            return TCGADataset(self.rna_train_file, self.gcn_train_file, self.dna_train_file)
        elif partition == "val":
            return TCGADataset(self.rna_val_file, self.gcn_val_file, self.dna_val_file)
        elif partition == "predict":
            return TCGADataset(self.rna_predict_file, self.gcn_predict_file, self.dna_predict_file)


class TCGADataset(Dataset):
    """
    Dataset Wrapper of TCGA Data (so data is loaded only once)
    """

    def __init__(self, rna_data, gcn_data, dna_data):
        """
        Args:
            rna_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of RNA-seq
            gcn_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of Gene Copy Number
            dna_data (numpy ndarray): Numpy array of the same shape as the .csv holding float32 data of DNA Methylation
        """
        assert rna_data.shape[0] == gcn_data.shape[0] == dna_data.shape[0], "Datasets do not have equal samples"

        self.rna_data = rna_data
        self.gcn_data = gcn_data
        self.dna_data = dna_data

    def __len__(self):
        return self.rna_data.shape[0]

    def __getitem__(self, idx):
        return self.rna_data[idx], self.gcn_data[idx], self.dna_data[idx]
