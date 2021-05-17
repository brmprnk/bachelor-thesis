from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class MVAE(nn.Module):
    def __init__(self, latent_dim=100, use_cuda=False):
        super(MVAE, self).__init__()
        # define q(z|x_i) for i = 1...2
        self.rna_encoder = Encoder(latent_dim, 3)
        self.gcn_encoder = Encoder(latent_dim, 1)

        # define p(x_i|z) for i = 1...2
        self.rna_decoder = Decoder(latent_dim, 3)
        self.gcn_decoder = Decoder(latent_dim, 1)

        # define q(z|x) = q(z|x_1)...q(z|x_6)
        self.experts = ProductOfExperts()
        self.latent_dim = latent_dim
        self.use_cuda = use_cuda

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:  # return mean during inference
            return mu

    def forward(self, rna=None, gcn=None):
        mu, logvar = self.get_params(rna=rna, gcn=gcn)

        # re-parameterization trick to sample
        z = self.reparameterize(mu, logvar)

        # reconstruct inputs based on sample
        rna_recon = self.rna_decoder(z)
        gcn_recon = self.gcn_decoder(z)

        return rna_recon, gcn_recon, mu, logvar

    def get_params(self, rna=None, gcn=None):
        # define universal expert
        batch_size = get_batch_size(rna, gcn)
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.latent_dim), use_cuda=use_cuda)

        if rna is not None:
            rna_mu, rna_logvar = self.rna_encoder(rna)
            mu = torch.cat((mu, rna_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, rna_logvar.unsqueeze(0)), dim=0)

        if gcn is not None:
            gcn_mu, gcn_logvar = self.gcn_encoder(gcn)
            mu = torch.cat((mu, gcn_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, gcn_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine Gaussian's
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


def get_batch_size(*args):
    for arg in args:
        if arg is not None:
            return arg.size(0)


class Encoder(nn.Module):
    """Parametrizes q(z|x).

    We will use this for every q(z|x_i) for all i.

    @param latent_dim: integer
                      number of latent dimensions
    @param n_channels: integer [default: 3]
                       number of input channels
    """

    def __init__(self, latent_dim, n_channels=3):
        super(Encoder, self).__init__()

        latent_dim = 100
        input_size = 5000
        hidden_dims = [256]

        modules = []
        if hidden_dims is None:
            hidden_dims = [256]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_size, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        self.latent_dim = latent_dim

    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)


        return mu, log_var


class Decoder(nn.Module):
    """Parametrizes p(x|z).

    We will use this for every p(x_i|z) for all i.

    @param latent_dim: integer
                      number of latent dimensions
    @param n_channels: integer [default: 3]
                       number of input channels
    """

    def __init__(self, latent_dim, n_channels=3):
        super(Decoder, self).__init__()
        latent_dim = 100
        input_size = 5000
        hidden_dims = [256]

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU()
        )
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_size),
            nn.Sigmoid())

    def forward(self, z):
        # the input will be a vector of size |latent_dim|
        result = self.decoder(z)
        result = self.final_layer(result)

        return result


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
