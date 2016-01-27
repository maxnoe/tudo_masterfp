import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.integrate import quad
from functools import partial
import uncertainties as unc


def linear(x, a, b, x0):
    return a * (x - x0) + b


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def poly(x, a0, a1, a2):
    return a2 * x**2 + a1 * x * a0

def diese_funktion_aus_der_anleitung(row, data, T_star = 318.15):
    mask = (data['T'] <= T_star) & (data['T'] >= row['T'])
    integral = np.trapz(data['I_corrected'][mask], data['T'][mask])
    return np.log(integral / row['I_corrected'])


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

    data['I_corrected'] = data['I'] - func(data['T'] - T0, *params)

    # px = np.linspace(220, 320, 1000)
    #
    # #plot current
    # plt.figure()
    # plt.plot(px, func(px - T0, *params))
    # plt.plot(data['T'], data['I'], '+', ms=4, label='Nicht berücksichtigt', color="#949494")
    # plt.plot(fit1['T'], fit1['I'], '+', ms=4, label='Fit-Region')
    # plt.legend(loc='upper left')
    # plt.xlabel(r'$T \mathrel{/} \si{\kelvin}$')
    # plt.ylabel(r'$I \mathrel{/} \si{\pico\ampere}$')
    # plt.ylim(0, 3200)
    # plt.tight_layout(pad=0)
    # plt.savefig('build/fit_non_linear.pdf')
    #
    #
    # #plot corrected current
    # plt.figure()
    # plt.plot(
    #     data['T'],
    #     data['I_corrected'],
    #     '+',
    #     ms=4,
    # )
    # plt.xlabel(r'$T \mathrel{/} \si{\kelvin}$')
    # plt.ylabel(r'$I \mathrel{/} \si{\pico\ampere}$')
    # plt.tight_layout(pad=0)
    # plt.savefig('build/fit_non_linear_corr.pdf')


    #fit activaiton energy
    T_max = data['I_corrected'].argmax()
    print('T_max = {} Kevins'.format(data['T'][T_max]))

    data = data.drop_duplicates(subset=['T'])
    f = partial(diese_funktion_aus_der_anleitung, data=data, T_star=310)
    data['activation'] = data.apply(f, axis=1)


    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data['T_inv'] = 1/ data['T']
    # data = data.query('(T > 275)')

    fit_data =data.query('(0.0033 < T_inv < 0.00345)')
    ignored_data = data[~data.index.isin(fit_data.index)]
    func = partial(linear, x0=fit_data['T_inv'].mean())
    params, cov = curve_fit(
        func, fit_data['T_inv'], fit_data['activation'],
    )
    a, b = unc.correlated_values(params, cov)

    print('Activation Energy {} Joule and {} eV'.format(a*const.k, a*const.k/const.e))

    with open('build/activation_work_fit.tex', 'w') as f:
        f.write('W = ')
        f.write(SI( a * const.k, r'\joule'))
        f.write(' = ')
        f.write(SI( a * const.k / const.e, r'\electronvolt'))

    plt.figure()
    plt.plot(fit_data['T_inv'], fit_data['activation'], '+', ms=4)
    plt.plot(ignored_data['T_inv'], ignored_data['activation'], '+', ms=4, color='#626262')
    plt.plot(fit_data['T_inv'], func(fit_data['T_inv'], *params), color='darkgray')
    plt.xlabel(r'$T^{-1} \mathrel{/} \si{\per\kelvin}$')
    plt.savefig('build/activation_energy.pdf')





if __name__ == '__main__':
    main()
