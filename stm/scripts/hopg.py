from read_nid import read_nid
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

from scipy.optimize import minimize

def distances(p, coordinates):
    return np.sqrt((p[0] - coordinates[:,0])**2 + (p[1] - coordinates[:,1])**2)


def grid_constants(coordinates):
    g_means = []
    g_stds = []
    for p in coordinates:
        # print(p)
        ds = distances(p, coordinates)
        #take 5 best distances for graphene
        g = ds[np.argsort(ds)[1:6]]
        g_means.append(g.mean())
        g_stds.append(g.std())

    return np.array(g_means),  np.array(g_stds)



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
    # embed()
    res = minimize(
        plane_chisq,
        [0.0001, 0.0001, -0.01],
        args=(x.ravel(), y.ravel(), data[1].ravel())
    )

    corrected = data[1] - plane(x, y, *res.x)

    coordinates = peak_local_max(
        rescale_intensity(corrected),
        min_distance=7,
    )
    print(coordinates)

    coordinates = coordinates * width / corrected.shape[0]
    g, err = grid_constants(coordinates)
    g_string = '\SI{{{:.4f} \pm {:.5f}}}{{\\nano\\meter}}'.format(g.mean(), err.mean())
    with open('build/grid_constant.tex', 'w') as f:
        f.write(g_string)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    plot = ax.pcolormesh(x, y, corrected, cmap='inferno')
    ax.plot(coordinates[:, 1], coordinates[:, 0], '.', ms=2.5)
    fig.colorbar(plot, ax=ax)
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)

    fig.savefig('build/plots/hopg.pdf')
