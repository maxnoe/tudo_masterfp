import numpy as np
from configparser import ConfigParser


def read_nid(filename):

    metadata = ConfigParser()
    with open(filename, 'rb') as f:

        found_empty_lines = 0
        lines = []
        while found_empty_lines < 2:
            line = f.readline()
            if line == b'\r\n':
                found_empty_lines += 1
            else:
                found_empty_lines = 0

            if not line.startswith(b'--'):
                lines.append(line)
        rawdata = f.read()

    metadata.read_string(b''.join(lines).decode('latin1'))

    data = np.frombuffer(rawdata, dtype='float16')
    N = np.sqrt((data.shape[0] - 1) / 4)

    return metadata, data[1:].reshape((4, N, N))


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

