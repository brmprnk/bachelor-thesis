# DNA model specification

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from utils import Constants
from .vae import VAE

scale_factor = 10000

# Classes
class Enc(nn.Module):

    def __init__(self, data_dim, latent_dim, num_hidden_layers, hidden_dim): #added hidden_dim
        super(Enc, self).__init__()

        self.input_size = data_dim
        self.latent_dim = latent_dim

        modules = []
        hidden_dim = [256]

        for h_dim in hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Linear(self.input_size, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )

        self.enc = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1], latent_dim)
    
    def read_count(self, x):
        read = torch.sum(x, axis=1)
        read = read.repeat(self.input_size, 1).t()
        return(read)

    def forward(self, x):
        read = self.read_count(x)
        x = x / read * scale_factor

        result = self.enc(x)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result).clamp(-12, 12)  # restrict to avoid torch.exp() over/underflow
        return mu, F.softmax(log_var, dim=-1) * log_var.size(-1) + Constants.eta

class Dec(nn.Module):
    """ Generate an MNIST image given a sample from the latent space. """

    def __init__(self, data_dim, latent_dim, num_hidden_layers, hidden_dim): #added hidden_dim
        super(Dec, self).__init__()
        self.input_size = data_dim
        modules = []
        
        hidden_dim = [256]

        modules.append(nn.Sequential(
            nn.Linear(latent_dim, hidden_dim[-1]),
            nn.BatchNorm1d(hidden_dim[-1]),
            nn.ReLU()
        ))

        self.dec = nn.Sequential(*modules)

        self.fc31 = nn.Linear(hidden_dim[-1], self.input_size)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim[-1], self.input_size),
            nn.Sigmoid())

    def forward(self, z):
        d = self.dec(z)
        # print("FROM my decoder I get")
        log_r = self.fc31(d).clamp(-12,12) #restrict to avoid torch.exp() over/underflow
        r = torch.exp(log_r)

        # IMPORTANT
        # This clamp messes with the result values, a lot.
        # Perhaps normalize data to be between 0.00000001 (=Constants.eps) and 0.99999999 instead of just clamping to
        # fit. With negative values in the data (or data not normalized), this leads to wildly inconsistent results

        # p_clamped = self.final_layer(d).clamp(Constants.eps, 1 - Constants.eps)  # restrict to avoid probs = 0,1
        p_not_clamped = self.final_layer(d)
        return r, p_not_clamped


class DNA(VAE):
    """ Derive a specific sub-class of a VAE for DNA. """

    def __init__(self, params):
        super(DNA, self).__init__(
            dist.Laplace,
            dist.Normal, #likelihood
            dist.Laplace,
            Enc(params.r_dim, params.latent_dim, params.num_hidden_layers, params.r_hidden_dim),
            Dec(params.r_dim, params.latent_dim, params.num_hidden_layers, params.r_hidden_dim),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'dna'
        self.data_dim = self.params.r_dim
        self.llik_scaling = 1.

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)
    
    def forward(self, x):
        read_count = self.enc.read_count(x)
        self._qz_x_params = self.enc(x)
        # print("HI I AM IN MY VAE")
        # print("FROM encoder I get", self._qz_x_params[0].size(), self._qz_x_params[1].size())
        qz_x = self.qz_x(*self._qz_x_params)
        # print("THEN I DO POSTERIOR DISTRIBUTION")
        # print(qz_x)
        zs = qz_x.rsample()
        r, _ = self.dec(zs)
        r = r / scale_factor * read_count 
        px_z = self.px_z(r, _)
        return qz_x, px_z, zs
