from read_nid import read_nid
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == '__main__':
    metadata, data = read_nid('data/hopg.nid')

    fig, axs = plt.subplots(2, 2)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, 0.75 * width)


    current_label = r'$I \mathbin{/} \si{\pico\ampere}$'
    height_label = r'$z \mathbin{/} \si{\nano\meter}$'

    for i in range(2):
        for j in range(2):
            divider = make_axes_locatable(axs[i][j])
            cax = divider.append_axes("right", size="5%", pad=0.02)

            key = 'DataSet-{}:{}'.format(i, j)
            width = metadata.getfloat(key, 'dim0range') * 1e9
            height = metadata.getfloat(key, 'dim1range') * 1e9
            scale = metadata.getfloat(key, 'dim2range')
            if j == 0:
                scale *= 1e12
            else:
                scale *= 1e9

            if j == 1:
                data[key] -= data[key].min()

            p = axs[j][i].imshow(
                data[key] * scale,
                cmap='inferno',
                extent=(0, width, 0, height),
                interpolation='nearest',
            )
            p.set_rasterized(True)
            fig.colorbar(
                p, cax=cax,
                label=current_label if i == 0 else height_label,
            )
            axs[j][i].grid(False)

    axs[0][0].set_xticklabels([])
    axs[0][0].set_ylabel(r'$y \mathbin{/} \si{\nano\meter}$')
    axs[0][0].set_title('up')

    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[0][1].set_xlabel('')
    axs[0][1].set_ylabel('')
    axs[0][1].set_title('down')

    axs[1][0].set_ylabel(r'$y \mathbin{/} \si{\nano\meter}$')
    axs[1][0].set_xlabel(r'$x \mathbin{/} \si{\nano\meter}$')

    axs[1][1].set_xlabel(r'$x \mathbin{/} \si{\nano\meter}$')
    axs[1][1].set_yticklabels([])

    fig.tight_layout(pad=0)
    fig.savefig('build/plots/hopg_uncorrected.pdf')

