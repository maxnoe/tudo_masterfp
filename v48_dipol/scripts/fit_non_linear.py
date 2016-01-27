import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit

from functools import partial
import uncertainties as unc
from uncertainties import unumpy as unp

from pint import UnitRegistry
u = UnitRegistry()


def linear(x, a, b, x0):
    return a * (x - x0) + b


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def poly(x, a0, a1, a2):
    return a2 * x**2 + a1 * x * a0


def diese_funktion_aus_der_anleitung(row, data, T_star=318.15):
    mask = (data['T'] <= T_star) & (data['T'] >= row['T'])
    integral = np.trapz(data['I_corrected'][mask], data['T'][mask])
    return np.log(integral / row['I_corrected'])


def SI(value, unit):
    result = r'\SI{{{:.2}}}{{{}}}'.format(value, unit)
    result = result.replace('(', '').replace(')', '').replace('+/-', r'\pm')
    return result


def num(value):
    result = r'\num{{{:.2}}}'.format(value)
    result = result.replace('(', '').replace(')', '').replace('+/-', r'\pm')
    return result


def main():
    data = pd.read_csv(
        'data/messwerte_2.csv',
        skiprows=(1, 2, 3, 4, 5),
    )
    data['T'] = data['T'].apply(const.C2K)

    func = lambda x, a, b: linear(x, a, b, x0=data['t'].mean())
    params, cov = curve_fit(func, data['t'], data['T'])
    heating_rate, _ = unc.correlated_values(params, cov)
    heating_rate *= u.kelvin / u.minute
    print('Rate = {}'.format(heating_rate))
    plt.plot(data['t'], data['T'], '+', ms=2)
    plt.plot(data['t'], func(data['t'], *params),
             label='Ausgleichsgerade', color='gray')
    plt.xlabel(r'$t \mathrel{/} \si{\second}$')
    plt.ylabel(r'$T \mathrel{/} \si{\kelvin}$')
    plt.tight_layout(pad=0)
    plt.savefig('build/heating_rate.pdf')

    fit1 = data.query('(245 < T < 270) | (T > 310)')

    T0 = fit1['T'].min()

    func = exponential
    params = 10, 0.1, data['I'].min()
    params, cov = curve_fit(
        func, fit1['T'] - T0, fit1['I'],  params
    )
    data['I_corrected'] = data['I'] - func(data['T'] - T0, *params)

    px = np.linspace(220, 320, 1000)

    # plot current
    plt.figure()
    plt.plot(px, func(px - T0, *params))
    plt.plot(data['T'], data['I'], '+', ms=4,
             label='Nicht berücksichtigt', color="#949494")
    plt.plot(fit1['T'], fit1['I'], '+', ms=4, label='Fit-Region')
    plt.legend(loc='upper left')
    plt.xlabel(r'$T \mathrel{/} \si{\kelvin}$')
    plt.ylabel(r'$I \mathrel{/} \si{\pico\ampere}$')
    plt.ylim(0, 3200)
    plt.tight_layout(pad=0)
    plt.savefig('build/fit_non_linear.pdf')

    # plot corrected current
    plt.figure()
    plt.plot(
        data['T'],
        data['I_corrected'],
        '+',
        ms=4,
    )
    plt.xlabel(r'$T \mathrel{/} \si{\kelvin}$')
    plt.ylabel(r'$I \mathrel{/} \si{\pico\ampere}$')
    plt.tight_layout(pad=0)
    plt.savefig('build/fit_non_linear_corr.pdf')

    # fit activaiton energy
    T_max = data['T'][data['I_corrected'].argmax()] * u.kelvin
    print('T_max = {} '.format(T_max))

    data = data.drop_duplicates(subset=['T'])
    f = partial(diese_funktion_aus_der_anleitung, data=data, T_star=310)
    data['activation'] = data.apply(f, axis=1)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data['T_inv'] = 1 / data['T']
    # data = data.query('(T > 275)')

    fit_data = data.query('(0.0033 < T_inv < 0.00345)')
    ignored_data = data[~data.index.isin(fit_data.index)]
    func = partial(linear, x0=fit_data['T_inv'].mean())
    params, cov = curve_fit(
        func, fit_data['T_inv'], fit_data['activation'],
    )
    a, b = unc.correlated_values(params, cov)
    W = a * u.kelvin * u.boltzmann_constant
    print('Activation Energy {} / {}'.format(
        W.to(u.joule), W.to(u.eV))
    )

    relaxation_time = (((u.boltzmann_constant * T_max**2) /
                        (W * heating_rate)) *
                       unp.exp(- W.to(u.joule).magnitude /
                               (const.k * T_max.magnitude))
                       )

    print(relaxation_time.to(u.second))

    tau_0 = relaxation_time / \
        unp.exp(W.to(u.joule).magnitude / (const.k * T_max.magnitude))
    print('Tau 0 : {}'.format(tau_0))

    with open('build/activation_work_fit.tex', 'w') as f:
        f.write('W = ')
        f.write(SI(W.to('J').magnitude, r'\joule'))
        f.write(' = ')
        f.write(SI(W.to('eV').magnitude, r'\electronvolt'))

    with open('build/tau.tex', 'w') as f:
        f.write('\\tau(T_{max}) = ')
        f.write(SI(relaxation_time.to('s').magnitude, r'\second'))

    with open('build/tau_0.tex', 'w') as f:
        f.write('\\tau_0 = ')
        f.write(SI(tau_0.to('s').magnitude, r'\second'))

    plt.figure()
    plt.plot(fit_data['T_inv'], fit_data['activation'], '+', ms=4)
    plt.plot(ignored_data['T_inv'], ignored_data[
             'activation'], '+', ms=4, color='#626262')
    plt.plot(fit_data['T_inv'], func(
        fit_data['T_inv'], *params), color='darkgray')
    plt.xlabel(r'$T^{-1} \mathrel{/} \si{\per\kelvin}$')
    plt.savefig('build/activation_energy.pdf')


if __name__ == '__main__':
    main()
