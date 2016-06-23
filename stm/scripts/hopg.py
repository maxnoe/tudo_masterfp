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
    width = metadata.getfloat('DataSet-0:1', 'dim0range') * 1e9

    height_range = metadata.getfloat('DataSet-0:1', 'dim2range')
    height = data['DataSet-0:1'].astype('float64') * height_range * 1e9

    points = np.linspace(0, width, height.shape[0])
    x, y = np.meshgrid(points, points)

    res = minimize(
        plane_chisq,
        [0.0, -0.25, height.mean()],
        args=(x.ravel(), y.ravel(), height.ravel()),
        # tol=1e-15,
    )

    a, b, c = res.x
    print(res.x)
    alpha_x = np.arctan(a)
    alpha_y = np.arctan(b)
    print(alpha_x, alpha_y)

    coordinates = peak_local_max(
        rescale_intensity(corrected),
        min_distance=6,
    )
    height = height - plane(x, y, *res.x)

    x = x / np.cos(alpha_x)
    y = y / np.cos(alpha_y)
    print(1 / np.cos(alpha_x), 1 / np.cos(alpha_y))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    plot = ax.pcolormesh(x, y, height, cmap='viridis')
    # ax.plot(coordinates[:, 1], coordinates[:, 0], '.', ms=3)
    fig.colorbar(plot, ax=ax)
    # ax.set_xlim(0, width)
    # ax.set_ylim(0, width)
    plt.show()

    fig.savefig('build/plots/hopg.pdf')
