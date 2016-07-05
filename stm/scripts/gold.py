import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import correlated_values

from utils import load_corrected_image
from read_nid import read_nid


if __name__ == '__main__':

    key = 'DataSet-1:1'
    metadata, data = read_nid('data/Noethe_Bruegge_053_gold1.nid')
    width = metadata.getfloat(key, 'dim0range') * 1e9

    height, y, x = load_corrected_image(
        metadata, data, 'DataSet-1:1',
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    lower = np.percentile(height, 5)
    upper = np.percentile(height, 95)

    plot = ax.pcolormesh(x, y, height, cmap='inferno', vmin=lower, vmax=upper)
    plot.set_rasterized(True)

    fig.colorbar(plot, ax=ax, label=r'$z \mathbin{/} \si{\nano\meter}$')

    ax.set_xlim(0, x.max())
    ax.set_ylim(0, y.max())
    ax.plot(
        x[50, 0:2],
        np.linspace(150, 350, 2),
        'g-',
    )

    ax.set_xlabel(r'$x \mathbin{/} \si{\nano\meter}$')
    ax.set_ylabel(r'$y \mathbin{/} \si{\nano\meter}$')

    fig.tight_layout(pad=0)
    fig.savefig('build/plots/gold.pdf')

    def linear(x, a, b):
        return a * (x - 260) + b

    profile_height = height[50]
    profile_x = y[50]

    mask = (profile_x > 150) & (profile_x < 350)

    p1 = (210, 255)
    p2 = (265, 300)

    def build_mask(x, p):
        return (x >= p[0]) & (x <= p[1])

    mask1 = build_mask(profile_x, p1)
    mask2 = build_mask(profile_x, p2)

    params1, cov1 = curve_fit(linear, profile_x[mask1], profile_height[mask1])
    params2, cov2 = curve_fit(linear, profile_x[mask2], profile_height[mask2])
    # params3, cov3 = curve_fit(linear, profile_x[mask3], profile_height[mask3])
    a1, b1 = correlated_values(params1, cov1)
    a2, b2 = correlated_values(params2, cov2)
    # a3, b3 = correlated_values(params3, cov3)
    print(a1, b1)
    print(a2, b2)
    # print(a3, b3)

    with open('build/fitresults.tex', 'w') as f:
        lines = [
            r'\begin{align}',
            r'm_1 &= \num{{{:0.3f} +- {:0.3f}}} & b_1 &= \SI{{{:0.3f} +- {:0.3f}}}{{\nano\meter}} \\'.format(a1.n, a1.s, b1.n, b1.s),
            r'm_2 &= \num{{{:0.3f} +- {:0.3f}}} & b_2 &= \SI{{{:0.3f} +- {:0.3f}}}{{\nano\meter}}'.format(a2.n, a2.s, b2.n, b2.s),
            r'\end{align}',
        ]
        f.write('\n'.join(lines) + '\n')

    a = (a1 + a2) / 2

    delta_h = unp.cos(unp.arctan(a)) * (b1 - b2)
    print(delta_h)
    with open('build/height.tex', 'w') as f:
        f.write(r'\SI{{{:.1f} +- {:.1f}}}{{\pico\meter}}'.format(
            delta_h.n * 1000, delta_h.s * 1000)
        )
        f.write('\n')


    grid_constant_gold = 407.82
    step_factor =  np.sqrt(3)*delta_h*1000/grid_constant_gold
    print(step_factor)
    with open('build/step_factor.tex', 'w') as f:
        f.write(r'\num{{{:.1f} +- {:.1f}}}'.format(
            step_factor.n, step_factor.s)
        )
        f.write('\n')

    fig, ax = plt.subplots()
    ax.plot(profile_x[mask], profile_height[mask], lw=0.5)
    ax.plot(profile_x[mask1], linear(profile_x[mask1], *params1))
    ax.plot(profile_x[mask2], linear(profile_x[mask2], *params2))
    # ax.plot(profile_x[mask3], linear(profile_x[mask3], *params3))
    ax.set_xlabel(r'$y \mathbin{/} \si{\nano\meter}$')
    ax.set_ylabel(r'$z \mathbin{/} \si{\nano\meter}$')
    fig.tight_layout(pad=0)
    fig.savefig('build/plots/height_profile.pdf')
