from read_nid import read_nid
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial


if __name__ == '__main__':

    key = 'DataSet-1:1'
    metadata, data = read_nid('data/gold.nid')
    width = metadata.getfloat(key, 'dim0range') * 1e9

    height_range = metadata.getfloat(key, 'dim2range')
    height = data[key] * height_range * 1e9
    height = height - np.percentile(height, 5)

    points = np.linspace(0, width, height.shape[0])
    y, x = np.meshgrid(points, points)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, aspect=1)
    # lower = np.percentile(height, 5)
    # upper = np.percentile(height, 95)

    # plot = ax.pcolormesh(x, y, height, cmap='viridis', vmin=lower, vmax=upper)

    # fig.colorbar(plot, ax=ax)

    # ax.set_xlim(0, x.max())
    # ax.set_ylim(0, y.max())
    # # fig.savefig('build/plots/hopg.pdf')
    # plt.show()

    def linear(x, a, b):
        return a * (x - 110) + b

    profile_height = height[50]
    profile_x = y[50]

    mask = (profile_x > 150) & (profile_x < 350)
    mask1 = (profile_x > 170) & (profile_x < 255)
    mask2 = (profile_x > 265) & (profile_x < 330)
    profile_x -= 150

    (a1, b1), cov1 = curve_fit(linear, profile_x[mask1], profile_height[mask1])
    (a2, b2), cov2 = curve_fit(linear, profile_x[mask2], profile_height[mask2])
    print(a1, b1)
    print(a2, b2)

    print(np.cos(np.arctan(a1)) * (b1 - b2))

    plt.plot(profile_x[mask], profile_height[mask])
    plt.plot(profile_x[mask1], linear(profile_x[mask1], a1, b1))
    plt.plot(profile_x[mask2], linear(profile_x[mask2], a2, b2))
    plt.show()
