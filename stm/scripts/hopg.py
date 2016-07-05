from read_nid import read_nid
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from uncertainties import ufloat
from uncertainties import umath
from uncertainties import correlated_values
from skimage.feature import peak_local_max

from scipy.optimize import curve_fit

from utils import load_corrected_image


def rad2deg(angle):
    return angle / np.pi * 180


def line(x, m, b):
    return m*x + b


def distances(p, coordinates):
    return np.sqrt((p[0] - coordinates[:, 0])**2 + (p[1] - coordinates[:, 1])**2)


def get_next_point_on_diagonal(p, coordinates):
    ds = distances(p, coordinates)
    cs = coordinates[np.argsort(ds)[0:5]]
    bs = cs[(cs[:, 0] > p[0]) & (cs[:, 1] > p[1])]
    if bs.shape[0] > 1:
        return bs[0]
    else:
        return bs.ravel()


def get_next_point_on_horizontal(p, coordinates):
    ds = distances(p, coordinates)
    cs = coordinates[np.argsort(ds)[0:10]]
    # print('-------')
    # print('p:')
    # print(p)
    # print(cs)
    # print(cs[:,0] - p[0])
    # print((np.abs(cs[:,0] - p[0]) < 0.02))
    bs = cs[(np.abs(cs[:, 0] - p[0]) < 0.03) & (cs[:, 1] > p[1])]
    # print('selected:')
    # print(bs)
    if bs.shape[0] > 1:
        return bs[0]
    else:
        return bs.ravel()


def grid(path, name, key):
    metadata, data = read_nid(path)

    height, x, y = load_corrected_image(metadata, data, key=key)

    peaks = peak_local_max(
        height,
        min_distance=6,
    ).astype('float64')

    peaks[:, 0] = peaks[:, 0] * x.max() / height.shape[0]
    peaks[:, 1] = peaks[:, 1] * y.max() / height.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    plot = ax.pcolormesh(x, y, height, cmap='inferno')
    plot.set_rasterized(True)

    ax.plot(peaks[:, 1], peaks[:, 0], 'w.', ms=3)
    fig.colorbar(plot, ax=ax, label=r'$z \mathbin{/} \si{\nano\meter}$')

    # print(peaks)

    d1 = np.array([0, 0])
    diagonal_points = []
    for a in range(13):
        d1 = get_next_point_on_diagonal(d1, peaks)
        if len(d1) == 0:
            break
        diagonal_points.append(d1)

    diagonal_distances = np.array([
        distance.euclidean(a, b)
        for a, b in zip(diagonal_points[:-1], diagonal_points[1:])]
    )
    gd = ufloat(diagonal_distances.mean(), diagonal_distances.std())
    print('Gitterkonstante a_1: {}'.format(gd))

    diagonal_points = np.array(diagonal_points)
    x_coords = diagonal_points[:, 1]
    y_coords = diagonal_points[:, 0]
    popt, pcov = curve_fit(line, x_coords, y_coords)
    m, b = correlated_values(popt, pcov)
    xs = np.linspace(0, 2.44, 50)
    ax.plot(xs, line(xs, m.n, b.n), color='lightgray')
    angle_diagonal = umath.atan(m)
    # print(angle_diagonal)

    for d in diagonal_points:
        ax.plot(d[1], d[0], 'o', color='#46d7ff', alpha=0.7)

    d1 = [0.33,  0]
    horizontal_points = []
    for a in range(10):
        d1 = get_next_point_on_horizontal(d1, peaks)
        if len(d1) == 0:
            break
        horizontal_points.append(d1)

    for d in horizontal_points:
        ax.plot(d[1], d[0], 'o', color='#dfec56', alpha=0.7)

    horizontal_points = np.array(horizontal_points)
    x_coords = horizontal_points[:, 1]
    y_coords = horizontal_points[:, 0]
    popt, pcov = curve_fit(line, x_coords, y_coords)
    m, b = correlated_values(popt, pcov)
    angle_horizontal = umath.atan(m)
    angle = angle_diagonal - angle_horizontal

    # should be 60
    correction_factor = np.tan(np.pi/3)/umath.tan(angle)
    print('Gemessener Winkel ist {}. Soll ist 60. Korrekturfaktor ist {}'.format(
        rad2deg(angle), correction_factor
    ))
    ax.plot(xs, line(xs, m.n, b.n), color='lightgray')

    horizontal_distances = np.array([
        distance.euclidean(a, b)
        for a, b in zip(horizontal_points[:-1], horizontal_points[1:])
    ])
    gh = ufloat(horizontal_distances.mean(), horizontal_distances.std())
    print('Gitterkonstante a_2: {}'.format(gh))

    ax.set_xlim(0, x.max())
    ax.set_ylim(0, y.max())

    g_string = '\SI{{{:.3f} \pm {:.3f}}}{{\\nano\\meter}}'.format(gh.n, gh.s)
    with open('build/grid_constant_horizontal_{}.tex'.format(name), 'w') as f:
        f.write(g_string)

    g_string = r'$\num{{{:.2f} \pm {:.2f}}}$'.format(correction_factor.n, correction_factor.s)
    with open('build/correction_factor_{}.tex'.format(name), 'w') as f:
        f.write(g_string)


    g_string = '\SI{{{:.3f} \pm {:.3f}}}{{\\nano\\meter}}'.format(gd.n, gd.s)
    with open('build/grid_constant_diagonal_{}.tex'.format(name), 'w') as f:
        f.write(g_string)

    angle = angle_diagonal - angle_horizontal
    g_string = '\\ang{{{:.2f} \pm {:.2f}}}'.format(np.rad2deg(angle.n), np.rad2deg(angle.s))
    with open('build/grid_angle_{}.tex'.format(name), 'w') as f:
        f.write(g_string)

    # plt.show()
    ax.set_xlabel(r'$x \mathbin{/} \si{\nano\meter}$')
    ax.set_ylabel(r'$y \mathbin{/} \si{\nano\meter}$')
    fig.tight_layout(pad=0)
    fig.savefig('build/plots/hopg_{}.pdf'.format(name))


if __name__ == '__main__':
    print('up')
    grid('./data/Noethe_Bruegge_049_hopg1.nid', 'up', 'DataSet-0:1')
    print('down')
    grid('./data/Noethe_Bruegge_049_hopg1.nid', 'down', 'DataSet-1:1')
