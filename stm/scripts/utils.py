from scipy.optimize import minimize
import numpy as np


def plane(x, y, a, b, c):
    z = a*x + b*y + c
    return z


def plane_chisq(params, x, y, z):

    plane_z = plane(x, y, *params)
    chisq = np.sum((z - plane_z)**2)

    return chisq


def slope(x, y, axis=0):
    m = np.mean(y, axis=axis)
    return (m.max() - m.min()) / (x.max() - x.min())


def load_corrected_image(metadata, data, key='DataSet-0:1', x0=None):
    width = metadata.getfloat(key, 'dim0range') * 1e9

    height_range = metadata.getfloat(key, 'dim2range')
    height = data[key].astype('float64') * height_range * 1e9

    points = np.linspace(0, width, height.shape[0])
    x, y = np.meshgrid(points, points)
    # embed()
    print(repr(x0))
    if x0 is None:
        x0 = [slope(x, height, 0), slope(y, height, 1), height.mean()]
    print(x0)

    res = minimize(
        plane_chisq,
        x0=x0,
        args=(x.ravel(), y.ravel(), height.ravel()),
        # tol=1e-15,
    )

    a, b, c = res.x
    alpha_x = np.arctan(a)
    alpha_y = np.arctan(b)

    x = x / np.cos(alpha_x)
    y = y / np.cos(alpha_y)

    height = height - plane(x, y, *res.x)
    return height, x, y
