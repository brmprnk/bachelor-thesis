import torch
from torch.utils.data import Subset, DataLoader
from torchnet.dataset import TensorDataset

import umap
import umap.plot
from main import load_checkpoint
import datasets
import numpy as np
import pandas as pd
import sys

if __name__ == "__main__":
    print("-----   Loading Trained Model   -----")
    model_path = "/Users/bram/rp-group-21-bpronk/experiments/cancer3_kl_weight_0.0000001_meanre/17-06-2021 20:16:39ng9shsws"
    model = load_checkpoint(model_path + "/best_model.pth.tar", use_cuda=False)
    model.eval()

    cancer3types = False
    if cancer3types:
        print("Umap of 3 cancer type model")
        train_dataset = [
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_training.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_training.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_training.csv",
                            index_col=0).to_numpy())
        ]

        val_dataset = [
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_validation.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_validation.csv",
                            index_col=0).to_numpy()),
            np.float32(
                pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_validation.csv",
                            index_col=0).to_numpy())
        ]

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

        print(train_dataset[0].shape)
        print(val_dataset[1].shape)
        rna_data = np.vstack((train_dataset[0], val_dataset[0], predict_dataset[0]))
        gcn_data = np.vstack((train_dataset[1], val_dataset[1], predict_dataset[1]))
        dna_data = np.vstack((train_dataset[2], val_dataset[2], predict_dataset[2]))

        labels_train = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_training_3types.npy")
        labels_val = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_validation_3types.npy")
        labels_predict = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_predict_3types.npy")
        cancer_types_samples = np.concatenate((labels_train, labels_val, labels_predict))

        full_dataset = [torch.from_numpy(rna_data), torch.from_numpy(gcn_data), torch.from_numpy(dna_data)]

        # 1 batch for all data
        data_loader = torch.utils.data.DataLoader(TensorDataset(full_dataset), batch_size=len(cancer_types_samples), shuffle=False)

        # Fetch reconstruction loss
        moe_rna_rna = np.load("{}/Recon array rna_rna.npy".format(model_path))
        moe_gcn_gcn = np.load("{}/Recon array gcn_gcn.npy".format(model_path))
        moe_dna_dna = np.load("{}/Recon array dna_dna.npy".format(model_path))
        losses = [np.round(moe_rna_rna[-1], 4), np.round(moe_gcn_gcn[-1], 4), np.round(moe_dna_dna[-1], 4)]
        recon_loss = np.round(sum(losses) / len(losses), 4)



        title = "Mixture of Experts 3 Cancer Types (BRCA, KIRC, LUAD) Latent Space (KL Weighing): \nEpochs : 100, Batch size : 256, Latent space : 128, LR: 0.0001, KL_Weight: 0.0000001\n Reconstruction loss : {}".format(recon_loss)
        save_dir = "/Users/bram/Desktop/moe_kl_umap_3cancer"
        save_file = "{}/UMAP Mixture of Experts 3 Cancer Types - 128 latentdim - 0.0001 LR - 0.0000001 KL_scalar.png".format(save_dir)
        background="black"
        color_key_cmap="Paired"

    else:
        cancer_types_samples = np.load("/Users/bram/Desktop/CSE3000/data/shuffle_cancertype/shuffle_cancertypes_samples.npy")
        indices_path = model_path

        # Load Data
        tcga_data = datasets.TCGAData(indices_path=indices_path)
        rna_dataset = tcga_data.get_data_partition("RNA")
        gcn_dataset = tcga_data.get_data_partition("GCN")
        dna_dataset = tcga_data.get_data_partition("DNA")

        print("len of rna_dataset = ", len(rna_dataset))

        print(np.arange(len(rna_dataset.data)))

        full_dataset = [Subset(rna_dataset.data, np.arange(len(rna_dataset.data))),
                        Subset(gcn_dataset.data, np.arange(len(gcn_dataset.data))),
                        Subset(dna_dataset.data, np.arange(len(dna_dataset.data)))]

        print(full_dataset)
        print(len(full_dataset[0]))

        # 1 batch for all data
        # noinspection PyTypeChecker
        # PyTypeChecker is invalid here, TensorDataSet is fine too for a DataLoader object
        data_loader = DataLoader(TensorDataset(full_dataset), batch_size=len(rna_dataset), shuffle=False)


        title = "Mixture of Experts Latent Space (KL Weighing): \nEpochs : 100, Batch size : 256, Latent space : 128, LR: 0.0001, KL_Weight: 0.0000001\n Reconstruction loss : {}".format(recon_loss)
        save_dir = "/Users/bram/Desktop/moe_kl_umap_3cancer"
        save_file = "{}/UMAP Mixture of Experts 3 Cancer Types - 128 latentdim - 0.0001 LR - 0.0000001 KL_scalar.png".format(save_dir)
        background = "white"
        color_key_cmap="Spectral"


    # Get Latent Space
    print("Entering Data into model, fetching Z and fitting UMAP.")
    z = 0
    latents = []
    device = torch.device("cpu")
    for idx, dataT in enumerate(data_loader):

        data = [d.to(device) for d in dataT]
        print(data)
        qz_xs, px_zs, zss = model(data)
        print(zss)
        for el in zss:
            print(el.size())
        print(len(zss))

        latents = model.latents(data)
        print("nr of latents", len(latents))

        z = sum(latents) / len(latents)
        z = z.cpu().detach().numpy()

    mapper = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean'
    ).fit(z)

    # mapper_rna = umap.UMAP(
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     n_components=2,
    #     metric='euclidean'
    # ).fit(latents[0])
    #
    # mapper_gcn = umap.UMAP(
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     n_components=2,
    #     metric='euclidean'
    # ).fit(latents[1])
    #
    # mapper_dna = umap.UMAP(
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     n_components=2,
    #     metric='euclidean'
    # ).fit(latents[2])


    print("Umap has been fit. Now plot")

    # p_rna = umap.plot.points(mapper_rna, labels=cancer_types_samples)
    # umap.plot.plt.title(
    #     "Mixture of Experts RNA-seq only: \nEpochs : 100, Batch size : 256, Latent space : 128, LR: 0.0001\n Umap (n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidian')"
    # )
    # umap.plot.plt.savefig("/Users/bram/Desktop/RNA-seq Z 3modal clamped.png", dpi=400)
    #
    # p_gcn = umap.plot.points(mapper_gcn, labels=cancer_types_samples)
    # umap.plot.plt.title(
    #     "Mixture of Experts GCN only: \nEpochs : 100, Batch size : 256, Latent space : 128, LR: 0.0001\n Umap (n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidian')"
    # )
    # umap.plot.plt.savefig("/Users/bram/Desktop/GCN Z 3modal clamped.png", dpi=400)
    #
    # p_dna = umap.plot.points(mapper_dna, labels=cancer_types_samples)
    # umap.plot.plt.title(
    #     "Mixture of Experts DNAme only: \nEpochs : 100, Batch size : 256, Latent space : 128, LR: 0.0001\n Umap (n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidian')"
    # )
    # umap.plot.plt.savefig("/Users/bram/Desktop/DNAme Z 3modal clamped.png", dpi=400)

    p = umap.plot.points(mapper, labels=cancer_types_samples, color_key_cmap=color_key_cmap, background=background)
    umap.plot.plt.title(title)
    # umap.plot.plt.legend()
    # umap.plot.plt.show()
    umap.plot.plt.savefig(save_file, dpi=800)

