import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes, mark_inset
)

plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect='equal')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_xlim(-0.02, 1.0)
ax.set_ylim(0, 1.02)
ax_zoom = zoomed_inset_axes(ax, 3, loc=4)
ax_zoom.set_xticks([0, 0.1, 0.2])
ax_zoom.set_yticks([0.8, 0.9, 1])
ax_zoom.set_xlim(-0.01, 0.2)
ax_zoom.set_ylim(0.8, 1.01)
ax_zoom.xaxis.set_ticks_position('top')
ax_zoom.xaxis.set_label_position('top')

infiles = [
    './build/AdaBoost.hdf5',
    './build/RandomForest.hdf5',
    './build/ExtraTrees.hdf5',
    './build/NaiveBayes.hdf5',
]
names = [os.path.splitext(os.path.basename(f))[0] for f in infiles]

colors = ['b', 'r', 'c', 'k']
lines = []
for infile, color in zip(infiles, colors):
    perf = pd.read_hdf(infile)

    # df = perf.mean(axis=0)
    # line, = ax.plot(
    #     df['fpr'],
    #     df['tpr'],
    #     alpha=0.3,
    #     color=color,
    # )
    # ax_zoom.plot(
    #     df['fpr'],
    #     df['tpr'],
    #     alpha=0.3,
    #     color=color,
    # )
    for n_cv, df in perf.iteritems():
        line, = ax.plot(
            df['fpr'],
            df['tpr'],
            linewidth=0.5,
            alpha=0.5,
            color=color,
        )
        ax_zoom.plot(
            df['fpr'],
            df['tpr'],
            alpha=0.5,
            color=color,
        )
    lines.append(line)

mark_inset(
    ax, ax_zoom, loc1=1, loc2=3, fc='none', ec="1", alpha=0.6,
)
ax.legend(lines, names)
fig.tight_layout()
plt.show()
