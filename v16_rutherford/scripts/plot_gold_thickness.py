import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linear(x, m, b):
    return m * x + b


def find_maxima(folder):
    data = pd.read_csv(
        path.join('data', folder, 'pressures.csv'),
        comment='#',
    )

    template = path.join('data', folder, 'TEK{:04d}.CSV')

    maxima = np.zeros_like(data.index)

    for i, row in enumerate(data.itertuples()):
        pulse = pd.read_csv(
            template.format(row.filenum),
            skiprows=18,
            header=None,
            usecols=(3, 4),
            names=['t', 'U'],
        )

        pulse['smooth'] = pd.rolling_mean(pulse['U'], 50, center=True)
        maxima[i] = pulse['smooth'].max()

    data['max'] = maxima
    return data


def main():

    with_foil = find_maxima('with_foil')
    without_foil = find_maxima('without_foil')

    params_with, cov_with = curve_fit(linear, with_foil['p'], with_foil['max'])
    params_without, cov_without = curve_fit(
        linear, without_foil['p'], without_foil['max']
    )

    p = np.linspace(0, 350, 2)

    plt.plot('p', 'max', '+', data=with_foil, label='Mit Folie')
    plt.plot(p, linear(p, *params_with))
    plt.plot('p', 'max', '+', data=without_foil, label='Ohne Folie')
    plt.plot(p, linear(p, *params_without))
    plt.xlabel(r'$p \mathrel{/} \si{\milli\bar}$')
    plt.ylabel(r'$\bar{U}_\text{max} \mathrel{/} \si{\volt}$')
    plt.legend()
    plt.ylim(0, 70)
    plt.tight_layout(pad=0)
    plt.savefig('build/plots/gold_thickness.pdf')


if __name__ == '__main__':
    main()
