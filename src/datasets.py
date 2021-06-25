import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
        rna_file = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_RNA_3000MAD_cancertypeknown.csv"
        rna_data = pd.read_csv(rna_file, usecols=range(1, 3001))
        print("-----   RNA file read   -----")

        # Gene Copy Number
        gcn_file = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_GCN_3000MAD_cancertypeknown.csv"
        gcn_data = pd.read_csv(gcn_file, usecols=range(1, 3001))
        print("-----   GCN file read   -----")

        # DNA Methylation
        dna_file = "/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_clamped_3modal_DNA_3000MAD_cancertypeknown.csv"
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

            self.training_ids = sets[0]
            self.validation_ids = sets[1]
            self.predict_ids = sets[2]

            if save_dir is None:
                print("Error, no save path is given so indices for data splits could not be saved. Exiting program")
                sys.exit()

            # Save the indices taken for reproducibility
            np.save("{}/training_indices.npy".format(save_dir), self.training_ids)
            np.save("{}/validation_indices.npy".format(save_dir), self.validation_ids)
            np.save("{}/predict_indices.npy".format(save_dir), self.predict_ids)

        else:  # Use predefined indices
            print("Using Predefined split")
            self.training_ids = np.load("{}/training_indices.npy".format(indices_path))
            self.validation_ids = np.load("{}/validation_indices.npy".format(indices_path))
            self.predict_ids = np.load("{}/predict_indices.npy".format(indices_path))

        # Create data arrays
        self.rna_dataset = np.nan_to_num(np.float32(rna_data.to_numpy()))

        self.gcn_dataset = np.nan_to_num(np.float32(gcn_data.to_numpy()))

        self.dna_dataset = np.nan_to_num(np.float32(dna_data.to_numpy()))

    def get_data_partition(self, data_type):
        if data_type == "RNA":
            return RNADataset(self.rna_dataset, self.training_ids, self.validation_ids, self.predict_ids)
        elif data_type == "GCN":
            return GCNDataset(self.gcn_dataset, self.training_ids, self.validation_ids, self.predict_ids)
        elif data_type == "DNA":
            return DNADataset(self.dna_dataset, self.training_ids, self.validation_ids, self.predict_ids)


class RNADataset(Dataset):
    """RNA Landmarks dataset."""

    def __init__(self, rna_data, training_ids, validation_ids, predict_ids):
        self.data = rna_data
        self.training_ids = training_ids
        self.validation_ids = validation_ids
        self.predict_ids = predict_ids

        print("RNA SHAPE : ", self.data.shape)
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


class GCNDataset(Dataset):
    """GCN Landmarks dataset."""

    def __init__(self, gcn_data, training_ids, validation_ids, predict_ids):
        self.data = gcn_data
        self.training_ids = training_ids
        self.validation_ids = validation_ids
        self.predict_ids = predict_ids

        print("GCN SHAPE : ", self.data.shape)
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

class DNADataset(Dataset):
    """DNA Landmarks dataset."""

    def __init__(self, dna_data, training_ids, validation_ids, predict_ids):
        self.data = dna_data
        self.training_ids = training_ids
        self.validation_ids = validation_ids
        self.predict_ids = predict_ids

        print("DNA SHAPE : ", self.data.shape)
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
