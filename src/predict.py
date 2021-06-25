from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.utils.data import Subset, DataLoader
from torchnet.dataset import TensorDataset

import numpy as np
import pandas as pd

from main import load_checkpoint
import datasets
from sklearn.metrics import mean_squared_error

PREDICT_DATA_SPLIT = 0.2


def predict_loss(recon_data, input_data):
    """
    Computes mean squared error between reconstructed data (in Tensor form) and actual input data

    @param recon_data:
    @param input_data:
    @return: int MSE
    """
    if not isinstance(recon_data, (list, np.ndarray)):
        recon_data = recon_data.values

    return mean_squared_error(input_data, recon_data)


def predict(trained_model, dataloader):
    trained_model.eval()
    modal = ["rna", "gcn", "dna"]
    uni, cross = [0, 0, 0], [0, 0, 0, 0, 0, 0]

    for i, dataT in enumerate(dataloader):
        data = [d.to(torch.device("cpu")) for d in dataT]
        recons_mat = trained_model.reconstruct_sample(data)
        for e, recons_list in enumerate(recons_mat):
            for d, recon in enumerate(recons_list):
                print("Modalities {} -> {}".format(modal[e], modal[d]))
                if e == d:  # Unimodal
                    recon = recon.squeeze(0).cpu().detach().numpy()
                    uni[e] = recon
                if e != d:
                    recon = recon.squeeze(0).cpu().detach().numpy()

                    if modal[e] == "rna":
                        if modal[d] == "gcn":
                            cross[0] = recon
                        elif modal[d] == "dna":
                            cross[1] = recon
                    elif modal[e] == "gcn":
                        if modal[d] == "rna":
                            cross[2] = recon
                        elif modal[d] == "dna":
                            cross[3] = recon
                    if modal[e] == "dna":
                        if modal[d] == "rna":
                            cross[4] = recon
                        elif modal[d] == "gcn":
                            cross[5] = recon

    return uni[0], uni[1], uni[2], cross[0], cross[1], cross[2], cross[3], cross[4], cross[5]


if __name__ == "__main__":

    print("-----   Loading Trained Model   -----")
    model_path = "/Users/bram/rp-group-21-bpronk/experiments/cancer3/14-06-2021 09:19:24nxkc8r8_/best_model.pth.tar"
    model = load_checkpoint(model_path, use_cuda=False)
    model.eval()

    cancer3types = False
    if cancer3types:
        indices_path = "/Users/bram/rp-group-21-bpronk/experiments/cancer3/14-06-2021 09:19:24nxkc8r8_"
        predict_dataset = [
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_predict.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_predict.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_predict.csv",
                            index_col=0).to_numpy())
        ]

        rna_data = predict_dataset[0]
        gcn_data = predict_dataset[1]
        dna_data = predict_dataset[2]

        predict_dataset = [torch.from_numpy(rna_data), torch.from_numpy(gcn_data), torch.from_numpy(dna_data)]
    else:
        indices_path = "/Users/bram/rp-group-21-bpronk/experiments/shuffle/14-06-2021 08:26:13q6w0c0s5"
        # Load Data
        tcga_data = datasets.TCGAData(indices_path=indices_path)
        rna_dataset = tcga_data.get_data_partition("RNA")
        gcn_dataset = tcga_data.get_data_partition("GCN")
        dna_dataset = tcga_data.get_data_partition("DNA")


        # Get the rows from the np array
        rna_data = rna_dataset.data[rna_dataset.predict_ids]
        gcn_data = gcn_dataset.data[gcn_dataset.predict_ids]
        dna_data = dna_dataset.data[dna_dataset.predict_ids]

        predict_dataset = [Subset(rna_dataset.data, rna_dataset.predict_ids),
                           Subset(gcn_dataset.data, gcn_dataset.predict_ids),
                           Subset(dna_dataset.data, dna_dataset.predict_ids)]

    # noinspection PyTypeChecker
    # PyTypeChecker is invalid here, TensorDataSet is fine too for a DataLoader object
    predict_loader = DataLoader(TensorDataset(predict_dataset), batch_size=len(rna_data), shuffle=False)

    (rna_recon_rna, gcn_recon_gcn, dna_recon_dna,
     rna_recon_gcn, rna_recon_dna,
     gcn_recon_rna, gcn_recon_dna,
     dna_recon_rna, dna_recon_gcn) = predict(model, predict_loader)

    rna_rna = predict_loss(rna_recon_rna, rna_data)
    rna_gcn = predict_loss(rna_recon_gcn, gcn_data)
    rna_dna = predict_loss(rna_recon_dna, dna_data)

    gcn_gcn = predict_loss(gcn_recon_gcn, gcn_data)
    gcn_rna = predict_loss(gcn_recon_rna, rna_data)
    gcn_dna = predict_loss(gcn_recon_dna, dna_data)

    dna_dna = predict_loss(dna_recon_dna, dna_data)
    dna_rna = predict_loss(dna_recon_rna, rna_data)
    dna_gcn = predict_loss(dna_recon_gcn, gcn_data)

    print("Prediction loss for RNA given RNA = ", rna_rna)
    print("Prediction loss for GCN given RNA = ", rna_gcn)
    print("Prediction loss for DNA given RNA = ", rna_dna)

    print("Prediction loss for GCN given GCN = ", gcn_gcn)
    print("Prediction loss for RNA given GCN = ", gcn_rna)
    print("Prediction loss for DNA given GCN = ", gcn_dna)

    print("Prediction loss for DNA given DNA = ", dna_dna)
    print("Prediction loss for RNA given DNA = ", dna_rna)
    print("Prediction loss for GCN given DNA = ", dna_gcn)

    dt_string = "MoE 14-06-2021 08:26:13q6w0c0s5"
    file1 = open("{}/predict_results.txt".format(indices_path), "a")
    file1.write("GCN from RNA at {} ||| {}\n".format(dt_string, rna_gcn))
    file1.write("DNA from RNA at {} ||| {}\n".format(dt_string, rna_dna))
    file1.write("RNA from GCN at {} ||| {}\n".format(dt_string, gcn_rna))
    file1.write("DNA from GCN at {} ||| {}\n".format(dt_string, gcn_dna))
    file1.write("RNA from DNA at {} ||| {}\n".format(dt_string, dna_rna))
    file1.write("GCN from DNA at {} ||| {}\n".format(dt_string, dna_gcn))

    file1.write("RNA from RNA (unimodal) at {} ||| {}\n".format(dt_string, rna_rna))
    file1.write("GCN from GCN (unimodal) at {} ||| {}\n".format(dt_string, gcn_gcn))
    file1.write("DNA from DNA (unimodal) at {} ||| {}\n".format(dt_string, dna_dna))
    file1.close()
