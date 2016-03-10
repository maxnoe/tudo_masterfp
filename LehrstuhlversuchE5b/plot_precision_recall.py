import matplotlib.pyplot as plt
import pandas as pd
import os

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, aspect='equal')
ax2 = fig.add_subplot(1, 2, 2, aspect='equal')
ax1.set_ylabel('Precision')
ax2.set_ylabel('Recall')

for ax in (ax1, ax2):
    ax.set_xlabel('Prediction Threshold')
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.5, 1.02)

infiles = [
    './build/AdaBoost.hdf5',
    './build/RandomForest.hdf5',
    './build/NaiveBayes.hdf5',
]

colors = [elem['color'] for elem in plt.rcParams['axes.prop_cycle']]

for infile, color in zip(infiles, colors):
    performance = pd.read_hdf(infile)

    mean_precision = performance.mean(axis=0)['precision']
    std_precision = performance.std(axis=0)['precision']
    mean_recall = performance.mean(axis=0)['tpr']
    std_recall = performance.std(axis=0)['tpr']

    ax1.plot(
        performance.mean(axis=0)['threshold'],
        mean_precision,
        color=color,
        label=os.path.splitext(os.path.basename(infile))[0],
    )
    ax1.fill_between(
        performance.mean(axis=0)['threshold'],
        mean_precision - std_precision,
        mean_precision + std_precision,
        color=color,
        linewidth=0,
        alpha=0.3,
    )

    ax2.plot(
        performance.mean(axis=0)['threshold'],
        mean_recall,
        color=color,
        label=os.path.splitext(os.path.basename(infile))[0],
    )
    ax2.fill_between(
        performance.mean(axis=0)['threshold'],
        mean_recall - std_recall,
        mean_recall + std_recall,
        color=color,
        linewidth=0,
        alpha=0.3,
    )

ax1.legend(loc='lower right')
fig.tight_layout(pad=0)
fig.savefig('build/precision_recall.pdf', bbox_inches='tight')
