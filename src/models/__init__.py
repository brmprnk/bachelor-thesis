from .vae_rna import RNA as VAE_rna
from .vae_protein import Protein as VAE_protein
from .vae_gcn import GCN as VAE_gcn

from .mmvae_rna_protein import RNA_Protein as VAE_rna_protein
from .mmvae_rna_gcn import RNA_GCN as VAE_rna_gcn
from .mmvae_rna_gcn_dna import RNA_GCN_DNA as VAE_rna_gcn_dna


__all__ = [VAE_rna, VAE_protein, VAE_gcn,
           VAE_rna_protein, VAE_rna_gcn, VAE_rna_gcn_dna]
