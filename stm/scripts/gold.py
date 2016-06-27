from read_nid import read_nid
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
from uncertainties import ufloat, correlated_values
import uncertainties.unumpy as unp


if __name__ == '__main__':

    key = 'DataSet-1:1'
    metadata, data = read_nid('data/gold.nid')
    width = metadata.getfloat(key, 'dim0range') * 1e9

    height_range = metadata.getfloat(key, 'dim2range')
    height = data[key] * height_range * 1e9
    height = height - np.percentile(height, 5)

    points = np.linspace(0, width, height.shape[0])
    y, x = np.meshgrid(points, points)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    lower = np.percentile(height, 5)
    upper = np.percentile(height, 95)

    plot = ax.pcolormesh(x, y, height, cmap='viridis', vmin=lower, vmax=upper)
    plot.set_rasterized(True)

    fig.colorbar(plot, ax=ax, label=r'$z \mathbin{/} \si{\nano\meter}$')

    ax.set_xlim(0, x.max())
    ax.set_ylim(0, y.max())

    ax.set_xlabel(r'$x \mathbin{/} \si{\nano\meter}$')
    ax.set_ylabel(r'$y \mathbin{/} \si{\nano\meter}$')

    fig.tight_layout(pad=0)
    fig.savefig('build/plots/gold.pdf')

    def linear(x, a, b):
        return a * (x - 260) + b

    profile_height = height[50]
    profile_y = y[50]

    mask = (profile_y > 150) & (profile_y < 350)
    mask1 = (profile_y > 210) & (profile_y < 255)
    mask2 = (profile_y > 265) & (profile_y < 310)
    # profile_y -= 150

    params1, cov1 = curve_fit(linear, profile_y[mask1], profile_height[mask1])
    params2, cov2 = curve_fit(linear, profile_y[mask2], profile_height[mask2])
    a1, b1 = correlated_values(params1, cov1)
    a2, b2 = correlated_values(params2, cov2)
    print(a1, b1)
    print(a2, b2)

    with open('build/fitresults.tex', 'w') as f:
        lines = [
            r'\begin{align}',
            r'm_1 &= \num{{{:0.3f} +- {:0.3f}}} & b_1 &= \SI{{{:0.3f} +- {:0.3f}}}{{\nano\meter}} \\'.format(a1.n, a1.s, b1.n, b1.s),
            r'm_2 &= \num{{{:0.3f} +- {:0.3f}}} & b_2 &= \SI{{{:0.3f} +- {:0.3f}}}{{\nano\meter}}'.format(a2.n, a2.s, b2.n, b2.s),
            r'\end{align}',
        ]
        f.write('\n'.join(lines) + '\n')

    a = 0.5 * (a1 + a2)

    delta_h = unp.cos(unp.arctan(a)) * (b1 - b2)
    with open('build/height.tex', 'w') as f:
        f.write(r'\SI{{{:.1f} +- {:.1f}}}{{\pico\meter}}'.format(
            delta_h.n * 1000, delta_h.s * 1000)
        )
        f.write('\n')

    fig, ax = plt.subplots()
    ax.plot(profile_y[mask], profile_height[mask], lw=0.5)
    ax.plot(profile_y[mask1], linear(profile_y[mask1], *params1))
    ax.plot(profile_y[mask2], linear(profile_y[mask2], *params2))
    ax.set_xlabel(r'$y \mathbin{/} \si{\nano\meter}$')
    ax.set_ylabel(r'$z \mathbin{/} \si{\nano\meter}$')
    fig.tight_layout(pad=0)
    fig.savefig('build/plots/height_profile.pdf')
