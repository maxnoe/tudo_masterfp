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

    fit1 = data.query('(250 < T < 270) | (T > 310)')

    func = exponential
    # func = lambda x, a, b: exponential(x, a, b, fit1['T'].min()100, 5, 500r)
    params, cov = curve_fit(
        func, fit1['T'] - fit1['T'], fit1['I'], [12.5, 0.08, data['I'].min()]
    )
    print(params)
    params = 12.5, 0.08, data['I'].min()

    px = np.linspace(240, 320, 1000)
    plt.plot(px, func(px - fit1['T'].min(), *params))
    plt.plot(data['T'], data['I'], '+', ms=3)
    plt.plot(fit1['T'], fit1['I'], '+', ms=3)

    plt.xlabel(r'$T \mathrel{/} \si{\kelvin}$')
    plt.ylabel(r'$I \mathrel{/} \si{\pico\ampere}$')

    plt.ylim(0, 3200)

    plt.tight_layout(pad=0)
    plt.savefig('build/fit_non_linear.pdf')

    plt.figure()
    plt.plot(
        data['T'],
        data['I'] - func(data['T'] - fit1['T'].min(), *params),
    )
    plt.savefig('build/fit_non_linear.pdf')


if __name__ == '__main__':
    main()
