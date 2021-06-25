# libraries
import numpy as np
import matplotlib.pyplot as plt

# Globals
BAR_WIDTH = 0.3
NUM_DECIMALS = 4

# # Prediction loss with MMVAE's (CROSSMODAL)
# labels = ['RNA --> GCN', 'RNA --> DNA',
#           'GCN --> RNA', 'GCN --> DNA',
#           'DNA --> RNA', 'DNA --> GCN']
# x = np.arange(len(labels))

# # set heights of bars ( RNA_GCN - RNA_DNA - GCN_RNA - GCN_DNA - DNA_RNA - DNA_GCN)
# MOE_CROSS = [np.round(0.313908189535141, NUM_DECIMALS), np.round(0.4422634541988373, NUM_DECIMALS),
#              np.round(0.34717288613319397, NUM_DECIMALS), np.round(0.44349557161331177, NUM_DECIMALS),
#              np.round(0.3480072319507599, NUM_DECIMALS), np.round(0.3141525685787201, NUM_DECIMALS)]

# POE_CROSS = [np.round(0.0454573780298233, NUM_DECIMALS),   np.round(0.0906691923737526, NUM_DECIMALS),
#              np.round(0.023636458441615105, NUM_DECIMALS), np.round(0.09066056460142136, NUM_DECIMALS),
#              np.round(0.02375185117125511, NUM_DECIMALS),  np.round(0.04365384951233864, NUM_DECIMALS)]

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - BAR_WIDTH/2, MOE_CROSS, BAR_WIDTH, label='Mixture of Experts', color="#bc5090")
# rects2 = ax.bar(x + BAR_WIDTH/2, POE_CROSS, BAR_WIDTH, label='Product of Experts', color="#ffa600")

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Prediction Loss per Modality', fontweight='bold')
# ax.set_ylabel('Mean Squared Error', fontweight="bold")
# ax.set_title('Prediction Loss per Modality (Crossmodal)\n(latent_dim={}, batch_size={}, epochs={}, lr={})'
#              .format(128, 256, 100, 0.001))

# ax.set_xticks(x)
# ax.set_ylim([0, 0.52])
# ax.set_xticklabels(labels, fontsize=8)
# ax.legend()

# ax.bar_label(rects1, padding=3, fontsize=7)
# ax.bar_label(rects2, padding=3, fontsize=7)

# fig.tight_layout()


# save_dir = "/Users/bram/Desktop/cancer3_experiment"
# plt.savefig("{}/Prediction Loss Crossmodal 14 June cancer3.png".format(save_dir), dpi=600)
# plt.show()

# Prediction loss with MMVAE's (UNIMODAL)

labels = ['RNA --> RNA', 'GCN --> GCN', 'DNA --> DNA']
x = np.arange(len(labels))

# set heights of bars ( MoE - PoE)
MOE_UNIMODAL = [np.round(0.347678542137146, NUM_DECIMALS), np.round(0.31398168206214905, 4), np.round(0.4437776803970337, 4)]
POE_UNIMODAL = [np.round(0.023760493844747543, NUM_DECIMALS), np.round(0.04532989487051964, 4), np.round(0.08991391211748123, 4)]

fig, ax = plt.subplots()
rects1 = ax.bar(x - BAR_WIDTH/2, MOE_UNIMODAL, BAR_WIDTH, label='Mixture of Experts', color="#bc5090")
rects2 = ax.bar(x + BAR_WIDTH/2, POE_UNIMODAL, BAR_WIDTH, label='Product of Experts', color="#ffa600")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Prediction Loss per Modality', fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontweight="bold")
ax.set_title('Prediction Loss per Modality (Unimodal)\n(latent_dim={}, batch_size={}, epochs={}, lr={})'
             .format(128, 256, 100, 0.001))

x_axis = np.arange(len(labels))
ax.set_xticks(x_axis)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3, fontsize=7)
ax.bar_label(rects2, padding=3, fontsize=7)

fig.tight_layout()

ax.set_ylim([0, 0.52])

save_dir = "/Users/bram/Desktop/cancer3_experiment"
plt.savefig("{}/Prediction Loss Unimodal 14 June cancer3.png".format(save_dir), dpi=600)
plt.show()
