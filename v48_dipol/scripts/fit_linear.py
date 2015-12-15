import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit
import uncertainties as unc


def linear(x, a, b, x0):
    return a * (x - x0) + b


def SI(value, unit):
    result = r'\SI{{{}}}{{{}}}'.format(value, unit)
    result = result.replace('+/-', '+-')
    result = result.replace('(', '')
    result = result.replace(')', '')
    return result


def num(value):
    result = r'\num{{{}}}'.format(value)
    result = result.replace('+/-', '+-')
    result = result.replace('(', '')
    result = result.replace(')', '')
    return result


def main():
    data = pd.read_csv(
        'data/messwerte_2.csv',
        skiprows=(1, 2, 3, 4, 5),
    )
    data['T'] = data['T'].apply(const.C2K)
    data['invT'] = 1 / data['T']
    data['logI'] = np.log(data['I'])

    fit_region = data.query('0.00344 < invT < 0.003625')

    func = lambda x, a, b: linear(x, a, b, x0=fit_region.invT.mean())

    params, cov = curve_fit(func, fit_region['invT'], fit_region['logI'])
    a, b = unc.correlated_values(params, cov)

    with open('build/fit_parameters.tex', 'w') as f:
        f.write(r'a &= {}'.format(SI(a, r'\kelvin')))
        f.write(' \\\\ \n ')
        f.write(r'b &= {}'.format(num(b)))

    with open('build/activation_work.tex', 'w') as f:
        f.write('W = ')
        f.write(SI(- a * const.k, r'\joule'))
        f.write(' = ')
        f.write(SI(- a * const.k / const.e, r'\electronvolt'))

    px = np.linspace(3.335e-3, 3.7e-3, 2)

    plt.plot(px, func(px, a.n, b.n), label='Ausgleichsgerade')
    plt.plot(data.invT, data.logI, '+', ms=3, label='Nicht berücksichtigt', color="#949494")
    plt.plot(fit_region.invT, fit_region.logI, '+', ms=3, label='Fit-Region')
    plt.legend()

    plt.xlabel(r'$T^{-1} \mathrel{/} \si{\per\kelvin}$')
    plt.ylabel(r'$\ln(I \mathrel{/} \si{\pico\ampere})$')

    plt.xlim(0.0031, 0.00445)
    plt.ylim(6, 8.1)

    plt.tight_layout(pad=0)
    plt.savefig('build/fit_linear.pdf')


if __name__ == '__main__':
    main()
