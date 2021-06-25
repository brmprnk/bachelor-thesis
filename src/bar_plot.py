# libraries
import numpy as np
import matplotlib.pyplot as plt

# set width of bars
width = 0.3

labels = ['RNA-seq', 'Gene CN', 'DNA Methylation', 'Average']
x = np.arange(len(labels) - 1)

print(x)

# set heights of bars
mofa = [0.3211, 0.0721, 0.0821]
moe = [0.3215, 0.0721, 0.0821]
poe = [0.3288, 0.0721, 0.0821]

average_mofa = 0.5412
average_moe = 0.7801
average_poe = 0.2022

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mofa, width, label='MOFA+', color="tab:red")
rects2 = ax.bar(x, moe, width, label='MMVAE: Mixture of Experts')
rects3 = ax.bar(x + width, poe, width, label='MMVAE: Product of Experts')

rects4 = ax.bar(3 - width, average_mofa, width, color="tab:red", hatch="//")
rects5 = ax.bar(3, average_moe, width, hatch="//")
rects6 = ax.bar(3 + width, average_poe, width, hatch="//")


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Reconstruction Loss per type', fontweight='bold')
ax.set_ylabel('Mean Squared Error', fontweight="bold")
ax.set_title('Reconstruction Loss per Modality')

x_axis = np.arange(len(labels))
ax.set_xticks(x_axis)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3, fontsize=7)
ax.bar_label(rects2, padding=3, fontsize=7)
ax.bar_label(rects3, padding=3, fontsize=7)
ax.bar_label(rects4, padding=3, fontsize=7)
ax.bar_label(rects5, padding=3, fontsize=7)
ax.bar_label(rects6, padding=3, fontsize=7)

fig.tight_layout()


# # Set position of bar on X axis
# r1 = np.arange(len(mofa))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]

# Make the plot
# plt.bar(r1, mofa, color='#7f6d5f', width=barWidth, edgecolor='white', label='MOFA+')
# plt.bar(r2, moe, color='#557f2d', width=barWidth, edgecolor='white', label='Mixture of Experts')
# plt.bar(r3, poe, color='#2d7f5e', width=barWidth, edgecolor='white', label='Product of Experts')

# Add xticks on the middle of the group bars

# plt.xticks([r + barWidth for r in range(len(mofa))], )

# Create legend & Show graphic
# plt.legend()
plt.show()