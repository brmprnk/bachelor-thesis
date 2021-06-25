import torch
from sklearn.metrics import mean_squared_error

from models import mmvae_rna_gcn
from models import mmvae_rna_gcn_dna

from utils import kl_divergence


def loss(model, x, kld_weight, val=None) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

    :return: cock
    """
    # Reconstruction loss
    recons_loss = reconstruction_loss(model, x, val)

    recons_loss /= float(2)  # Account for number of modalities

    # KLD Loss
    qz_xs, px_zs, zss = model(x)
    lpx_zs, kld_losses = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        kld_losses.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            lpx_zs.append(lpx_z.sum(-1))

    # print("stacked kld losses", torch.stack(kld_losses).shape)
    # print(torch.stack(kld_losses))
    # print(torch.stack(kld_losses)[0].sum())
    # print(torch.stack(kld_losses)[1].sum())
    #
    # print(torch.stack(kld_losses).sum())
    # print("from vanillva VAE", torch.mean(-0.5 * torch.stack(kld_losses).sum()))

    kld_loss = torch.mean(0.5 * torch.stack(kld_losses).sum())

    # print(recons_loss)
    # # print(kld_weight)
    # print(-kld_loss)
    # print(-kld_loss.sum())
    # print(torch.mean(kld_loss))

    # Loss
    total_loss = recons_loss + kld_weight * kld_loss
    return {'loss': total_loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}


def reconstruction_loss(model, data, val=None):
    model.eval()

    modal = ["rna", "gcn", "dna"]

    rna_recon_rna = 0
    rna_recon_gcn = 0
    rna_recon_dna = 0

    gcn_recon_gcn = 0
    gcn_recon_rna = 0
    gcn_recon_dna = 0

    dna_recon_dna = 0
    dna_recon_rna = 0
    dna_recon_gcn = 0

    recons_mat = super(mmvae_rna_gcn_dna.RNA_GCN_DNA, model).reconstruct_sample(data)
    for r, recons_list in enumerate(recons_mat):
        # print(modal[r], len(recons_list))
        for o, recon in enumerate(recons_list):
            # print(modal[o])
            _data = data[r].cpu().detach().numpy()
            recon = recon.squeeze(0).cpu().detach().numpy()

            if modal[r] == "rna":
                if modal[o] == "rna":
                    rna_recon_rna = mean_squared_error(_data, recon)
                elif modal[o] == "gcn":
                    rna_recon_gcn = mean_squared_error(_data, recon)
                elif modal[o] == "dna":
                    rna_recon_dna = mean_squared_error(_data, recon)

            elif modal[r] == "gcn":
                if modal[o] == "gcn":
                    gcn_recon_gcn = mean_squared_error(_data, recon)
                elif modal[o] == "rna":
                    gcn_recon_rna = mean_squared_error(_data, recon)
                elif modal[o] == "dna":
                    gcn_recon_dna = mean_squared_error(_data, recon)

            elif modal[r] == "dna":
                if modal[o] == "dna":
                    dna_recon_dna = mean_squared_error(_data, recon)
                elif modal[o] == "rna":
                    dna_recon_rna = mean_squared_error(_data, recon)
                elif modal[o] == "gcn":
                    dna_recon_gcn = mean_squared_error(_data, recon)

    # print("RNA RECON RNA loss:", rna_recon_rna)
    # print("RNA RECON GCN loss:", rna_recon_gcn)
    # print("RNA RECON DNA loss:", rna_recon_dna)
    # print("GCN RECON GCN loss:", gcn_recon_gcn)
    # print("GCN RECON RNA loss:", gcn_recon_rna)
    # print("GCN RECON DNA loss:", rna_recon_gcn)
    # print("DNA RECON DNA loss:", dna_recon_dna)
    # print("DNA RECON RNA loss:", dna_recon_rna)
    # print("DNA RECON GCN loss:", dna_recon_gcn)

    #     def modality_loss(self, val, key):
    #         self.reconstruct_losses[key].append(val)

    rna_recon_loss = (rna_recon_rna + rna_recon_gcn + rna_recon_dna) / 3
    gcn_recon_loss = (gcn_recon_gcn + gcn_recon_rna + gcn_recon_dna) / 3
    dna_recon_loss = (dna_recon_dna + dna_recon_rna + dna_recon_gcn) / 3

    if val is not None:  # Update reconstruction loss meter
        val.modality_loss(rna_recon_loss + gcn_recon_loss + dna_recon_loss, "average")
        val.modality_loss(rna_recon_rna, "rna_rna")
        val.modality_loss(gcn_recon_gcn, "gcn_gcn")
        val.modality_loss(dna_recon_dna, "dna_dna")
        val.modality_loss(rna_recon_gcn, "rna_gcn")
        val.modality_loss(rna_recon_dna, "rna_dna")
        val.modality_loss(gcn_recon_rna, "gcn_rna")
        val.modality_loss(gcn_recon_dna, "gcn_dna")
        val.modality_loss(dna_recon_rna, "dna_rna")
        val.modality_loss(dna_recon_gcn, "dna_gcn")

    return rna_recon_loss + gcn_recon_loss + dna_recon_loss


# multi-modal elbo
def m_elbo_naive(model, x):
    """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED"""
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            lpx_zs.append(lpx_z.sum(-1))

    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.sum()


def m_elbo_naive_warmup(model, x, beta):
    """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED"""
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.vaes[d].llik_scaling
            lpx_zs.append(lpx_z.sum(-1))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - beta * torch.stack(klds).sum(0))
    return obj.sum()