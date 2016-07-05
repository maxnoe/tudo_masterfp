'''
Usage:
    read_nid <inputfile> [options]

Options:
    -o <outpufile>    Save plot to <outputfile>
'''

import numpy as np
from configparser import ConfigParser
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from docopt import docopt

label = r'${} \,/\, \mathrm{{{}}}$'


def read_nid(filename):

    with open(filename, 'rb') as f:
        raw_metadata, raw_data = f.read().split(b'#!')

    metadata_lines = raw_metadata.decode('latin1').splitlines()
    # remove lines starting with --
    metadata_str = '\n'.join(filter(lambda x: not x.startswith('--'), metadata_lines))
    metadata = ConfigParser()
    metadata.read_string(metadata_str)

    keys = [
        key for key in metadata.keys()
        if re.match('DataSet-[0-9]+:[0-9]+', key)
    ]

    data = {}
    bytesread = 0
    for key in keys:
        points = metadata.getint(key, 'points')
        lines = metadata.getint(key, 'lines')
        bits = metadata.getint(key, 'savebits')
        data[key] = np.frombuffer(
            raw_data,
            dtype='float{}'.format(bits),
            count=lines * points,
            offset=bytesread,
        ).reshape((lines, points)).astype('float64')
        bytesread += lines * points * bits // 8

    return metadata, data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    args = docopt(__doc__)

    metadata, data = read_nid(args['<inputfile>'])

    fig, axs = plt.subplots(2, 2)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, 0.75 * width)

    current_label = label.format('I', 'pA')
    height_label = label.format('z', 'nm')

    for i in range(2):
        for j in range(2):
            ax = axs[j][i]
            divider = make_axes_locatable(ax)
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
                data[key] -= np.nanmin(data[key])

            img = data[key] * scale
            p = ax.imshow(
                img,
                cmap='inferno',
                extent=(0, width, 0, height),
                interpolation='nearest',
                vmin=np.nanmin(img.ravel()),
                vmax=np.nanmax(img.ravel()),
            )
            p.set_rasterized(True)
            fig.colorbar(
                p, cax=cax,
                label=current_label if j == 0 else height_label,
            )
            ax.grid(False)

    axs[0][0].set_xticklabels([])
    axs[0][0].set_ylabel(label.format('y', 'nm'))
    axs[0][0].set_title('up')
    axs[0][0].set_title(metadata['DataSet-0:0']['Frame'])

    axs[0][1].set_xticklabels([])
    axs[0][1].set_yticklabels([])
    axs[0][1].set_xlabel('')
    axs[0][1].set_ylabel('')
    axs[0][1].set_title(metadata['DataSet-1:0']['Frame'])

    axs[1][0].set_xlabel(label.format('x', 'nm'))
    axs[1][0].set_ylabel(label.format('y', 'nm'))

    axs[1][1].set_xlabel(label.format('x', 'nm'))
    axs[1][1].set_yticklabels([])

    fig.tight_layout(pad=0)
    if args['-o']:
        fig.savefig(args['-o'])
    else:
        plt.show()
