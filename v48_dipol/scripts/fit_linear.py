import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit
import uncertainties as unc


A = np.pi * 1.5e-3**2


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
        'build/data_corrected.csv',
    )
    data['invT'] = 1 / data['T']
    data['j'] = data['I_corrected'] / A
    data['logj'] = np.log(data['j'])

    fit_region = data.query('0.00345 < invT < 0.0037')
    func = lambda x, a, b: linear(x, a, b, x0=fit_region.invT.mean())

    params, cov = curve_fit(func, fit_region['invT'], fit_region['logj'])
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
    plt.plot(data.invT, data.logj, '+', ms=3, label='Nicht berücksichtigt', color="#949494")
    plt.plot(fit_region.invT, fit_region.logj, '+', ms=3, label='Fit-Region')
    plt.legend()

    plt.xlabel(r'$T^{-1} \mathrel{/} \si{\per\kelvin}$')
    plt.ylabel(r'$\ln(I \mathrel{/} \si{\pico\ampere})$')

    plt.xlim(0.0031, 0.0038)
    # plt.ylim(6, 8.1)

    plt.tight_layout(pad=0)
    plt.savefig('build/fit_linear.pdf')


if __name__ == '__main__':
    main()
