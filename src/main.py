import argparse
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from tempfile import mkdtemp

from torch.utils.data import Subset, DataLoader
from torchnet.dataset import TensorDataset

import pandas as pd
import matplotlib.pyplot as plt

import models
import objectives
from utils import Timer, save_vars
import datasets
from models.mmvae_rna_gcn import RNA_GCN
from models.mmvae_rna_gcn_dna import RNA_GCN_DNA

import pandas as pd
import numpy as np
import torch
from torch import optim


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.epochs = 50
        self.values = []
        self.reconstruct_losses = {
            "average" : [],
            "rna_rna" : [],
            "gcn_gcn" : [],
            "dna_dna" : [],
            "rna_gcn" : [],
            "rna_dna" : [],
            "gcn_rna": [],
            "gcn_dna": [],
            "dna_rna": [],
            "dna_gcn": [],
        }

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(self.avg.item())

    def modality_loss(self, val, key):
        self.reconstruct_losses[key].append(val)


def save_checkpoint(state, lowest_loss, save_dir):
    """
    Saves a Pytorch model's state, and also saves it to a separate object if it is the best model (lowest loss) thus far

    @param state:       Python dictionary containing the model's state
    @param lowest_loss: Boolean check if the current checkpoint has had the best (lowest) loss so far
    @param save_dir:      String of the folder to save the model to
    @return: None
    """
    # Save checkpoint
    # torch.save(state, os.path.join(save_dir, filename))

    # If this is the best checkpoint (lowest loss) thus far, copy this model to a file named model_best
    if lowest_loss:
        print("Best epoch thus far (lowest loss) --> Saving to model_best")
        torch.save(state, os.path.join(save_dir, 'best_model.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)

    # Create class for arguments in getting model
    class ModelArgs(object):
        def __init__(self):
            self.latent_dim = checkpoint['latent_dim']
            self.r_dim = checkpoint['r_dim']
            self.num_hidden_layers = checkpoint['num_hidden_layers']
            self.r_hidden_dim = checkpoint['r_hidden_dim']
            self.learn_prior = False
            self.llik_scaling = 1.

    params = ModelArgs()

    model = RNA_GCN_DNA(params)

    model.load_state_dict(checkpoint['state_dict'])
    return model


def train(epoch, train_loss_meter, train_recon_loss_meter, train_kld_loss_meter, args_kl_weight):
    model.train()
    b_loss = 0
    progress_bar = tqdm(total=len(train_loader))
    for idx, dataT in enumerate(train_loader):

        if dataT[0].size()[0] == 1:
            continue

        # Refresh optimizer
        optimizer.zero_grad()

        data = [d.to(device) for d in dataT]

        if args_kl_weight == 1:
            kld_weight = len(data[0].cpu().detach().numpy()) / len(train_loader.dataset)
        else:
            kld_weight = args_kl_weight
        loss = objective(model, data, kld_weight)

        # loss = -objective(model, data)
        # loss.backward()
        # optimizer.step()
        #
        # b_loss += loss.item()

        train_loss_meter.update(loss['loss'], len(data[0].cpu().detach().numpy()))
        train_recon_loss_meter.update(loss['Reconstruction_Loss'], len(data[0].cpu().detach().numpy()))
        train_kld_loss_meter.update(loss['KLD'], len(data[0].cpu().detach().numpy()))

        # compute and take gradient step
        loss['loss'].backward()
        optimizer.step()

        # if args.print_freq > 0 and idx % args.print_freq == 0:
        #     print("iteration {:04d}: loss: {:6.3f}".format(idx, loss.item() / args.batch_size))
        progress_bar.update()

    progress_bar.close()
    # print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, b_loss / len(train_loader.dataset)))
    print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))
    print('====> Epoch: {}\tReconstruction Loss: {:.4f}'.format(epoch, train_recon_loss_meter.avg))
    print('====> Epoch: {}\tKLD Loss: {:.4f}'.format(epoch, train_kld_loss_meter.avg))


def test(epoch, val_loss_meter, val_recon_loss_meter, val_kld_loss_meter, args_kl_weight):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(val_loader):
            if dataT[0].size()[0] == 1:
                continue

            data = [d.to(device) for d in dataT]  # multimodal

            # loss = -s_objective(model, data)
            # b_loss += loss.item()

            if args_kl_weight == 1: 
                kld_weight = len(data[0].cpu().detach().numpy()) / len(val_loader.dataset)
            else:
                kld_weight = args_kl_weight

            val_loss = objective(model, data, kld_weight, val=val_recon_loss_meter)

            print("length of update meter n : ", len(data[0].cpu().detach().numpy()))
            val_loss_meter.update(val_loss['loss'], len(data[0].cpu().detach().numpy()))
            val_recon_loss_meter.update(val_loss['Reconstruction_Loss'], len(data[0].cpu().detach().numpy()))
            val_kld_loss_meter.update(val_loss['KLD'], len(data[0].cpu().detach().numpy()))

    print('====> Epoch: {:03d} Validation loss: {:.4f}'.format(epoch, b_loss / len(val_loader.dataset)))

    print('====> Epoch: {:03d} Validation Loss: {:.4f}'.format(epoch, val_loss_meter.avg))
    print('====> Epoch: {:03d} Validation Reconstruction Loss: {:.4f}'.format(epoch, val_recon_loss_meter.avg))
    print('====> Epoch: {:03d} Validation KLD Loss: {:.4f}'.format(epoch, val_kld_loss_meter.avg))
    return val_loss_meter.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scMM Hyperparameters')
    parser.add_argument('--experiment', type=str, default='test', metavar='E',
                        help='experiment name')
    parser.add_argument('--model', type=str, default='rna_gcn_dna', metavar='M',
                        help='model name (default: rna_gcn_dna)')
    parser.add_argument('--obj', type=str, default='loss', metavar='O',
                        help='objective to use (default: elbo)')
    parser.add_argument('--llik_scaling', type=float, default=1.,
                        help='likelihood scaling for cub images/svhn modality when running in'
                             'multimodal setting, set as 0 to use default value')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size for data (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='L',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--latent_dim', type=int, default=128, metavar='L',
                        help='latent dimensionality (default: 20)')
    parser.add_argument('--kl_weight', type=float, default=1,
                        help='Scalar for the KL term in the loss function (default: 1)')
    parser.add_argument('--num_hidden_layers', type=int, default=1, metavar='H',
                        help='number of hidden layers in enc and dec (default: 1)')
    parser.add_argument('--r_hidden_dim', type=int, default=256,
                        help='number of hidden units in enc/dec for gene')
    parser.add_argument('--p_hidden_dim', type=int, default=256,
                        help='number of hidden units in enc/dec for protein/peak')
    parser.add_argument('--pre_trained', type=str, default="",
                        help='path to pre-trained model (train from scratch if empty)')
    parser.add_argument('--learn_prior', action='store_true', default=False,
                        help='learn model prior parameters')
    parser.add_argument('--analytics', action='store_true', default=True,
                        help='disable plotting analytics')
    parser.add_argument('--print_freq', type=int, default=0, metavar='f',
                        help='frequency with which to print stats (default: 0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--r_dim', type=int, default=1)
    parser.add_argument('--p_dim', type=int, default=1)

    # args
    args = parser.parse_args()

    # Set CPU no GPU
    device = torch.device("cpu")

    # random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Current Time for output files
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")

    # set up run path
    experiment_dir = Path('../experiments/' + args.experiment)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=dt_string, dir=str(experiment_dir))
    print("Saving experiment in", runPath)

    # Load Data
    # Hack in a way for some different datasets to be tested, only 3 cancer types and pre-split
    cancer3types = False
    if cancer3types:
        print("Using a predefined split of 3 cancer types")
        train_dataset = [
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_training.csv", index_col=0).to_numpy())),
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_training.csv", index_col=0).to_numpy())),
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_training.csv", index_col=0).to_numpy()))
        ]

        val_dataset = [
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_validation.csv", index_col=0).to_numpy())),
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_validation.csv", index_col=0).to_numpy())),
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_validation.csv", index_col=0).to_numpy()))
        ]

        predict_dataset = [
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/rna_preprocess_3types_predict.csv", index_col=0).to_numpy())),
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/gcn_preprocess_3types_predict.csv", index_col=0).to_numpy())),
            torch.from_numpy(np.float32(pd.read_csv("/Users/bram/Desktop/CSE3000/data/3types/dna_preprocess_3types_predict.csv", index_col=0).to_numpy()))
        ]

        labels_train = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_training_3types.npy")
        labels_val = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_validation_3types.npy")
        labels_predict = np.load("/Users/bram/Desktop/CSE3000/data/3types/labels_predict_3types.npy")

        nr_of_samples = len(labels_train) + len(labels_val) + len(labels_predict)
        prediction_batch_size = len(labels_predict)
        args.r_dim = 3000
        args.p_dim = 3000

        print("All data loaded in, now creating dataloaders")
        train_loader = DataLoader(TensorDataset(train_dataset), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_dataset), batch_size=len(labels_val), shuffle=False)

    else:
        tcga_data = datasets.TCGAData(save_dir=runPath)
        rna_data = tcga_data.get_data_partition("RNA")
        gcn_data = tcga_data.get_data_partition("GCN")
        dna_data = tcga_data.get_data_partition("DNA")

        nr_of_samples = rna_data.shape[0]
        prediction_batch_size = len(rna_data.predict_ids)
        args.r_dim = rna_data.data.shape[1]
        args.p_dim = gcn_data.data.shape[1]

        # Split Data into train, validation & prediction
        train_dataset = [Subset(rna_data, rna_data.training_ids), Subset(gcn_data, gcn_data.training_ids),
                         Subset(dna_data, dna_data.training_ids)]
        val_dataset = [Subset(rna_data, rna_data.validation_ids), Subset(gcn_data, gcn_data.validation_ids),
                       Subset(dna_data, dna_data.validation_ids)]
        predict_dataset = [Subset(rna_data, rna_data.predict_ids), Subset(gcn_data, gcn_data.predict_ids),
                           Subset(dna_data, dna_data.predict_ids)]


        train_loader = DataLoader(TensorDataset(train_dataset), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_dataset), batch_size=len(rna_data.validation_ids), shuffle=False)

    total_batches = len(train_loader)

    # Load or get model
    pretrained_path = ""
    if args.pre_trained:
        pretrained_path = args.pre_trained
        pretrain_args = args
        # pretrain_args.learn_prior = False

        # Load model
        modelC = getattr(models, 'VAE_{}'.format(pretrain_args.model))
        model = modelC(pretrain_args).to(device)
        print('Loading model {} from {}'.format(model.modelName, pretrained_path))
        model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
        model._pz_params = model._pz_params
    else:
        # load model
        modelC = getattr(models, 'VAE_{}'.format(args.model))
        print(args)

        model = modelC(args).to(device)
        torch.save(args, runPath + '/args.rar')

    # preparation for training
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)
    objective = getattr(objectives, args.obj)
    s_objective = getattr(objectives, args.obj)

    # Log Data shape, input arguments and model
    model_file = open("{}/Model and args.txt".format(runPath), "a")
    model_file.write("Running at {}\n".format(dt_string))
    model_file.write("Input shape : {}, 3000\n".format(nr_of_samples))
    model_file.write("Input args : {}\n".format(args))
    model_file.write("Mixture of Experts Model : {}".format(model))
    model_file.close()

    # Now start running model
    with Timer('MM-VAE') as t:
        train_loss_meter = AverageMeter("Loss")
        train_recon_loss_meter = AverageMeter("Reconstruction Loss")
        train_kld_loss_meter = AverageMeter("KLD Loss")

        val_loss_meter = AverageMeter("Validation Loss")
        val_recon_loss_meter = AverageMeter("Validation Reconstruction Loss")
        val_kld_loss_meter = AverageMeter("Validation KLD Loss")

        for epoch in range(1, args.epochs + 1):
            train(epoch, train_loss_meter, train_recon_loss_meter, train_kld_loss_meter, args.kl_weight)

            latest_loss = test(epoch, val_loss_meter, val_recon_loss_meter, val_kld_loss_meter, args.kl_weight)
            save_vars(val_recon_loss_meter, runPath + '/losses.rar')

            # Save the model on the final epoch
            if epoch == args.epochs:
                print("Saving model to run Path")
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'best_loss': latest_loss,
                    'latent_dim': args.latent_dim,
                    'r_dim': args.r_dim,
                    'num_hidden_layers': args.num_hidden_layers,
                    'r_hidden_dim': args.r_hidden_dim,
                    'optimizer': optimizer.state_dict(),
                }, True, runPath)

    if args.analytics:
        def get_latent(dataloader, train_test, run_path):
            model.eval()
            with torch.no_grad():
                if args.model == 'rna_protein':
                    modal = ['rna', 'protein']
                elif args.model == 'rna_gcn':
                    modal = ['rna', 'gcn']
                elif args.model == 'rna_gcn_dna':
                    modal = ['rna', 'gcn', 'dna']
                pred = []
                for i, dataT in enumerate(dataloader):
                    data = [d.to(device) for d in dataT]
                    lats = model.latents(data, sampling=False)
                    if i == 0:
                        pred = lats
                    else:
                        for m, lat in enumerate(lats):
                            pred[m] = torch.cat([pred[m], lat], dim=0) 
            
                for m, lat in enumerate(pred):
                    lat = lat.cpu().detach().numpy()
                    lat = pd.DataFrame(lat)
                    lat.to_csv('{}/lat_{}_{}.csv'.format(run_path, train_test, modal[m]))

                print("Getting latents")
                print(len(pred))

                mean_lats = sum(pred)/len(pred)
                mean_lats = mean_lats.cpu().detach().numpy()
                mean_lats = pd.DataFrame(mean_lats)
                mean_lats.to_csv('{}/lat_{}_mean.csv'.format(run_path, train_test))

        def predict(dataloader, run_path):
            model.eval()
            with torch.no_grad():
                uni, cross = [], []
                for i, dataT in enumerate(dataloader):
                    data = [d.to(device) for d in dataT]
                    recons_mat = model.reconstruct_sample(data)
                    for e, recons_list in enumerate(recons_mat):
                        for d, recon in enumerate(recons_list):
                            if e == d:
                                recon = recon.squeeze(0).cpu().detach().numpy()
                                if i == 0:
                                    uni.append(recon)
                                else:
                                    uni[e] = np.vstack([uni[e], recon])
                            if e != d:
                                recon = recon.squeeze(0).cpu().detach().numpy()
                                if i == 0:
                                    cross.append(recon)
                                else:
                                    cross[e] = np.vstack([cross[e], recon])

                print("Length of uni : ", len(uni))
                print("Length of cross : ", len(cross))

                # Save for later reproducibility
                # pd.DataFrame(uni[0]).to_csv('{}/pred_rna_rna.csv'.format(run_path))
                # pd.DataFrame(uni[1]).to_csv('{}/pred_gcn_gcn.csv'.format(run_path))
                # pd.DataFrame(cross[0]).to_csv('{}/pred_rna_gcn.csv'.format(run_path))
                # pd.DataFrame(cross[1]).to_csv('{}/pred_gcn_rna.csv'.format(run_path))

        train_loader = DataLoader(TensorDataset(train_dataset), batch_size=args.batch_size, shuffle=False)

        predict_loader = DataLoader(TensorDataset(predict_dataset), batch_size=prediction_batch_size, shuffle=False)

        get_latent(train_loader, 'train', runPath)
        # get_latent(test_loader, 'test', runPath)

        predict(predict_loader, runPath)
        
        # model.traverse(runPath, device)

        # Save all reconstruction losses
        modal = ['rna', 'gcn', 'dna']
        for modal1 in modal:
            for modal2 in modal:
                key = "{}_{}".format(modal1, modal2)
                np.save("{}/Recon array {}.npy".format(runPath, key),
                        np.array(val_recon_loss_meter.reconstruct_losses[key]))


        # Do some plotting
        x_axis = [*range(1, args.epochs + 1)]

        # Loss
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        plt.plot(x_axis, train_loss_meter.values[::total_batches], marker='.', color='tab:purple')
        for x, y in zip(x_axis, train_loss_meter.values[::total_batches]):
            if x == 1 or x % 25 == 0:
                ax1.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title(
            "Mixture of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.latent_dim, args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Training Loss")
        plt.savefig("{}/Loss {}.png".format(runPath, dt_string), dpi=400)

        # Reconstruction Loss
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        plt.plot(x_axis, train_recon_loss_meter.values[::total_batches], marker='.', color='tab:orange')
        for x, y in zip(x_axis, train_recon_loss_meter.values[::total_batches]):
            if x == 1 or x % 25 == 0:
                ax2.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title(
            "Mixture of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.latent_dim, args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Training Reconstruction Loss")
        plt.savefig("{}/Recon Loss {}.png".format(runPath, dt_string), dpi=400)

        # KLD Loss
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        plt.plot(x_axis, train_kld_loss_meter.values[::total_batches], marker='.', color='tab:red')
        for x, y in zip(x_axis, train_kld_loss_meter.values[::total_batches]):
            if x == 1 or x % 25 == 0:
                ax3.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title(
            "Mixture of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.latent_dim, args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Training KLD Loss")
        plt.savefig("{}/KLD Loss {}.png".format(runPath, dt_string), dpi=400)

        # Validation Loss
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        plt.plot(x_axis, val_loss_meter.values[::1], marker='.', color='tab:purple')
        for x, y in zip(x_axis, val_loss_meter.values[::1]):
            if x == 1 or x % 25 == 0:
                ax4.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title(
            "Mixture of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.latent_dim, args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Validation Loss")
        plt.savefig("{}/Validation Loss {}.png".format(runPath, dt_string), dpi=400)

        # Validation Reconstruction Loss
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)
        plt.plot(x_axis, val_recon_loss_meter.values[::1], marker='.', color='tab:orange', label="Average")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["rna_rna"], marker='.', color='tab:blue',
                 label="rna_rna")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["gcn_gcn"], marker='.', color='tab:green',
                 label="gcn_gcn")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["dna_dna"], marker='.', color='tab:red',
                 label="dna_dna")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["rna_gcn"], marker='.', color='tab:purple',
                 label="rna_gcn")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["rna_dna"], marker='.', color='tab:brown',
                 label="rna_dna")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["gcn_rna"], marker='.', color='tab:pink',
                 label="gcn_rna")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["gcn_dna"], marker='.', color='tab:gray',
                 label="gcn_dna")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["dna_rna"], marker='.', color='tab:olive',
                 label="dna_rna")
        plt.plot(x_axis, val_recon_loss_meter.reconstruct_losses["dna_gcn"], marker='.', color='tab:cyan',
                 label="dna_gcn")
        for x, y in zip(x_axis, val_recon_loss_meter.values[::1]):
            if x == 1 or x % 25 == 0:
                ax5.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title(
            "Mixture of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.latent_dim, args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Validation Reconstruction Loss")
        plt.savefig("{}/Validation Recon Loss {}.png".format(runPath, dt_string), dpi=400)

        # Validation KLD Loss
        fig6 = plt.figure(6)
        ax6 = fig6.add_subplot(111)
        plt.plot(x_axis, val_kld_loss_meter.values[::1], marker='.', color='tab:red')
        for x, y in zip(x_axis, val_kld_loss_meter.values[::1]):
            if x == 1 or x % 25 == 0:
                ax6.annotate('  ({}, {:.4f})'.format(x, y), xy=(x, y), textcoords='data', fontsize=9)
        plt.title(
            "Mixture of Experts: RNA-seq and GCN (gistic2)\nLatent space : {}, LR: {}".format(args.latent_dim, args.lr))
        plt.xlabel("Epochs")
        plt.ylabel("Average Validation KLD Loss")
        plt.savefig("{}/Validation KLD Loss {}.png".format(runPath, dt_string), dpi=400)
