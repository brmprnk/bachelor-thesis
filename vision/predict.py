from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd

import torch
import math
from sklearn.metrics import mean_squared_error
from train import load_checkpoint

PREDICT_DATA_SPLIT = 0.2

def predict_loss(recon_data, input_data):
    """

    @param recon_data:
    @param input_data:
    @return: int MSE
    """
    recon_data = recon_data.to_numpy()
    input_data = input_data.values

    print(type(recon_data))
    print(type(input_data))
    print(recon_data.shape)
    print(input_data.shape)

    return mean_squared_error(input_data, recon_data)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=1,
                        help='Number of images and texts to sample [default: 1]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir('./samples'):
        os.makedirs('./samples')

    args.model_path = "/Users/bram/multimodal-vae-public-master/vision/trained_models/checkpoint.pth.tar"
    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model.eval()

    print("-----   RNA file read   -----")
    rna_file = "/Users/bram/Desktop/CSE3000/data/RNA_COMMONGNC.csv"
    rna_data = pd.read_csv(rna_file, usecols=range(1, 5001))
    num_predict_samples = math.ceil(PREDICT_DATA_SPLIT * rna_data.shape[0])
    rna_data_predict_split = rna_data.tail(num_predict_samples)
    print("RNA SHAPE : ", rna_data.shape)

    rna_sample = np.nan_to_num(np.float32(rna_data_predict_split.to_numpy()))

    print("-----   GCN file read   -----")
    gcn_file = "/Users/bram/Desktop/CSE3000/data/GNC_POSITIVE_COMMONRNA.csv.csv"
    gcn_data = pd.read_csv(gcn_file, usecols=range(1, 5001))
    num_predict_samples = math.ceil(PREDICT_DATA_SPLIT * gcn_data.shape[0])
    gcn_data_predict_split = gcn_data.tail(num_predict_samples)
    print("GCN SHAPE : ", gcn_data.shape)

    rna_sample = np.nan_to_num(np.float32(rna_data_predict_split.to_numpy()))
    gcn_sample = np.nan_to_num(np.float32(gcn_data_predict_split.to_numpy()))
    args.n_samples = rna_sample.shape[0]

    print(model)
    rna_sample = torch.Tensor(rna_sample)
    gcn_sample = torch.Tensor(gcn_sample)

    (rna_recon_rna, rna_recon_gcn, rna_mu, rna_logvar) = model(rna=rna_sample)
    (gcn_recon_rna, gcn_recon_gcn, gcn_mu, gcn_logvar) = model(gcn=gcn_sample)

    # # generate image and text
    # SIGMOID
    # image_recon = F.sigmoid(model.image_decoder(sample)).cpu().data
    # gray_recon = F.sigmoid(model.gray_decoder(sample)).cpu().data

    print(rna_recon_rna)
    print(type(rna_recon_rna))
    # print(rna_recon_rna.shape)

    rna_prediction_loss = predict_loss(gcn_recon_rna, rna_data_predict_split)
    gcn_prediction_loss = predict_loss(rna_recon_gcn, gcn_data_predict_split)

    print("Prediction loss for RNA = ", rna_prediction_loss)
    print("Prediction loss for GCN = ", gcn_prediction_loss)

    print("Total prediction loss = ", (rna_prediction_loss + gcn_prediction_loss) / 2)
