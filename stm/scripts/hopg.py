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

    height = height - plane(x, y, *res.x)
    coordinates = peak_local_max(
        rescale_intensity(height),
        min_distance=6,
    )

    x = x / np.cos(alpha_x)
    y = y / np.cos(alpha_y)

    coords_scaled = np.empty_like(coordinates, dtype=float)
    coords_scaled[:, 0] = coordinates[:, 0].astype(float) * x.max() / height.shape[0]
    coords_scaled[:, 1] = coordinates[:, 1].astype(float) * y.max() / height.shape[1]
    print(coords_scaled)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    plot = ax.pcolormesh(x, y, height, cmap='viridis')

    ax.plot(coords_scaled[:, 1], coords_scaled[:, 0], '.', ms=3)
    fig.colorbar(plot, ax=ax)

    ax.set_xlim(0, x.max())
    ax.set_ylim(0, y.max())
    plt.show()

    fig.savefig('build/plots/hopg.pdf')
