import torch

from sklearn.metrics import mean_squared_error
from train import load_checkpoint
import datasets


def predict_loss(recon_data, input_data):
    """
    Computes mean squared error between reconstructed data (in Tensor form) and actual input data

    @param recon_data:
    @param input_data:
    @return: int MSE
    """
    return mean_squared_error(input_data, recon_data)


if __name__ == "__main__":

    print("-----   Loading Trained Model   -----")
    model_path = "/Users/bram/multimodal-vae-public-master/vision/results/PoE 26-05-2021 08:20:17/with_prior.pth.tar"
    model = load_checkpoint(model_path, use_cuda=False)
    model.eval()

    indices_path = "/Users/bram/multimodal-vae-public-master/vision/results/PoE 26-05-2021 08:20:17"
    tcga_data = datasets.TCGAData(indices_path=indices_path)

    predict_dataset = tcga_data.get_data_partition("predict")

    # 1 batch for prediction
    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=len(predict_dataset), shuffle=False)

    # FOR 2 Predictions
    for batch_idx, (rna, gcn, dna) in enumerate(predict_loader):
        (joint_recon_rna, joint_recon_gcn, joint_recon_dna, joint_mu, joint_logvar) = model(dna=dna)

        predicted_rna = joint_recon_rna.detach().numpy()

    for batch_idx, (rna, gcn, dna) in enumerate(predict_loader):
        (joint_recon_rna, joint_recon_gcn, joint_recon_dna, joint_mu, joint_logvar) = model(dna=dna)

        predicted_gcn = joint_recon_gcn.detach().numpy()

    for batch_idx, (rna, gcn, dna) in enumerate(predict_loader):
        (joint_recon_rna, joint_recon_gcn, joint_recon_dna, joint_mu, joint_logvar) = model(gcn=gcn)

        predicted_dna = joint_recon_dna.detach().numpy()

    rna_loss = predict_loss(predicted_rna, predict_dataset.rna_data)
    gcn_loss = predict_loss(predicted_gcn, predict_dataset.gcn_data)
    dna_loss = predict_loss(predicted_dna, predict_dataset.dna_data)

    print("rna loss = ", rna_loss)
    print("gcn loss = ", gcn_loss)
    print("dna loss = ", dna_loss)

    # Output results
    result_dir = "/Users/bram/multimodal-vae-public-master/vision/predict_results"
    # Current Time from model for output files
    dt_string = "PoE 25-05-2021 17:12:45"

    # file3 = open("{}/3.txt".format(result_dir), "a")
    # file3.write("RNA Joint at {} ||| {}\n".format(dt_string, rna_loss))
    # file3.write("GCN Joint at {} ||| {}\n".format(dt_string, gcn_loss))
    # file3.write("DNA Joint at {} ||| {}\n\n".format(dt_string, dna_loss))
    # file3.close()
    #
    # file2 = open("{}/2.txt".format(result_dir), "a")
    # file2.write("RNA from others at {} ||| {}\n".format(dt_string, rna_loss))
    # file2.write("GCN from others at {} ||| {}\n".format(dt_string, gcn_loss))
    # file2.write("DNA from others at {} ||| {}\n\n".format(dt_string, dna_loss))
    # file2.close()
    #
    file1 = open("{}/1.txt".format(result_dir), "a")
    # file1.write("RNA from GCN at {} ||| {}\n".format(dt_string, rna_loss))
    file1.write("RNA from DNA at {} ||| {}\n".format(dt_string, rna_loss))
    # file1.write("GCN from RNA at {} ||| {}\n".format(dt_string, gcn_loss))
    file1.write("GCN from DNA at {} ||| {}\n".format(dt_string, gcn_loss))
    # file1.write("DNA from RNA at {} ||| {}\n".format(dt_string, dna_loss))
    file1.write("DNA from GCN at {} ||| {}\n".format(dt_string, dna_loss))

    # file1.write("RNA from RNA (unimodal) at {} ||| {}\n".format(dt_string, rna_loss))
    # file1.write("GCN from GCN (unimodal) at {} ||| {}\n".format(dt_string, gcn_loss))
    # file1.write("DNA from DNA (unimodal) at {} ||| {}\n".format(dt_string, dna_loss))
    file1.close()
