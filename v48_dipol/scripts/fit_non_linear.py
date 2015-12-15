import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit
import uncertainties as unc


def linear(x, a, b, x0):
    return a * (x - x0) + b


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def poly(x, a0, a1, a2):
    return a2 * x**2 + a1 * x * a0


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

    fit1 = data.query('(245 < T < 270) | (T > 310)')

    T0 = fit1['T'].min()

    func = exponential
    params = 10, 0.1, data['I'].min()
    params, cov = curve_fit(
        func, fit1['T'] - T0, fit1['I'],  params
    )
    print(params)

    px = np.linspace(220, 320, 1000)
    plt.plot(px, func(px - T0, *params))
    plt.plot(data['T'], data['I'], '+', ms=4)
    plt.plot(fit1['T'], fit1['I'], '+', ms=4)

    plt.xlabel(r'$T \mathrel{/} \si{\kelvin}$')
    plt.ylabel(r'$I \mathrel{/} \si{\pico\ampere}$')

    plt.ylim(0, 3200)

    plt.tight_layout(pad=0)
    plt.savefig('build/fit_non_linear.pdf')

    plt.figure()
    plt.plot(
        data['T'],
        data['I'] - func(data['T'] - T0, *params),
        '+',
        ms=4,
    )

    plt.xlabel(r'$T \mathrel{/} \si{\kelvin}$')
    plt.ylabel(r'$I \mathrel{/} \si{\pico\ampere}$')

    plt.tight_layout(pad=0)
    plt.savefig('build/fit_non_linear_corr.pdf')


if __name__ == '__main__':
    main()
