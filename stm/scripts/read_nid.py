import numpy as np
from configparser import ConfigParser
import re


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
        ).reshape((lines, points))
        bytesread += lines * points * bits // 8

    return metadata, data

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    metadata, data = read_nid(sys.argv[1])

    print(40 * '=')
    print('{string: ^40}'.format(string='Metadata'))
    print(40 * '=')
    for key, value in metadata['DataSet-Info'].items():
        print(key, value, sep=' = ')

    for i, image in enumerate(data, start=1):

        plt.subplot(2, 2, i)
        plt.imshow(image, cmap='viridis')
        plt.colorbar(ax=plt.gca())

    plt.show()
