from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import datasets

pca_rna = PCA(n_components=256)
pca_gcn = PCA(n_components=256)
pca_dna = PCA(n_components=256)

indices_path = "/Users/bram/multimodal-vae-public-master/vision/results/PoE 26-05-2021 08:20:17"
tcga_data = datasets.TCGAData(indices_path=indices_path)

train_dataset = tcga_data.get_data_partition("train")
val_dataset = tcga_data.get_data_partition("val")
predict_dataset = tcga_data.get_data_partition("predict")

pca_rna.fit(train_dataset.rna_data)
pca_gcn.fit(train_dataset.gcn_data)
pca_dna.fit(train_dataset.dna_data)

samples = len(predict_dataset.rna_data)

counting_rna = 0.0
counting_gcn = 0.0
counting_dna = 0.0

for index in range(samples):
    sample_rna, sample_gcn, sample_dna = predict_dataset.__getitem__(index)
    sample_rna = sample_rna.reshape(1, -1)  # Reshape (1, -1) if sample
    sample_gcn = sample_gcn.reshape(1, -1)  # Reshape (1, -1) if sample
    sample_dna = sample_dna.reshape(1, -1)  # Reshape (1, -1) if sample

    z_space_rna = pca_rna.transform(sample_rna)
    z_space_gcn = pca_rna.transform(sample_gcn)
    z_space_dna = pca_rna.transform(sample_dna)

    reconstructed_rna = pca_rna.inverse_transform(z_space_rna)
    reconstructed_gcn = pca_gcn.inverse_transform(z_space_gcn)
    reconstructed_dna = pca_dna.inverse_transform(z_space_dna)

    counting_rna += mean_squared_error(sample_rna, reconstructed_rna)
    counting_gcn += mean_squared_error(sample_gcn, reconstructed_gcn)
    counting_dna += mean_squared_error(sample_dna, reconstructed_dna)

average_mse_rna = counting_rna / samples
average_mse_gcn = counting_gcn / samples
average_mse_dna = counting_dna / samples

print("PCA loss RNA |||", average_mse_rna)
print("PCA loss GCN |||", average_mse_gcn)
print("PCA loss DNA |||", average_mse_dna)
