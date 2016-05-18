from read_nid import read_nid
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


def plane(x, y, a, b, c):
    z = a*x + b*y + c
    return z


def plane_chisq(params, x, y, z):

    plane_z = plane(x, y, *params)
    chisq = np.sum((z - plane_z)**2)

    return chisq


if __name__ == '__main__':

    metadata, data = read_nid('data/hopg.nid')
    width = float(metadata['DataSet-Info']['image size'].replace(',', '.')[:-2])

    points = np.linspace(0, width, data[1].shape[0])
    x, y = np.meshgrid(points, points)

    res = minimize(
        plane_chisq,
        [0.0001, 0.0001, -0.01],
        args=(x.ravel(), y.ravel(), data[1].ravel())
    )

    corrected = data[1] - plane(x, y, *res.x)

    coordinates = peak_local_max(
        rescale_intensity(corrected),
        min_distance=6,
    )

    coordinates = coordinates * width / corrected.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    plot = ax.pcolormesh(x, y, corrected, cmap='viridis')
    ax.plot(coordinates[:, 1], coordinates[:, 0], '.', ms=3)
    fig.colorbar(plot, ax=ax)
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)

    fig.savefig('build/plots/hopg.pdf')
