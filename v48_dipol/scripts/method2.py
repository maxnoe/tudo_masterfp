import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit

from functools import partial
import uncertainties as unc
from uncertainties import unumpy as unp

from pint import UnitRegistry
from IPython import embed
u = UnitRegistry()


def linear(x, a, b, x0):
    return a * (x - x0) + b


def diese_funktion_aus_der_anleitung(row, data, T_star=318.15):
    mask = (data['T'] <= T_star) & (data['T'] >= row['T'])
    integral = np.trapz(data['I_corrected'][mask], data['T'][mask])
    return np.log(integral / row['I_corrected'])


def SI(value, unit):
    result = r'\SI{{{:.3}}}{{{}}}'.format(value, unit)
    result = result.replace('(', '').replace(')', '').replace('+/-', r'\pm')
    return result


def num(value):
    result = r'\num{{{:.2}}}'.format(value)
    result = result.replace('(', '').replace(')', '').replace('+/-', r'\pm')
    return result


def main():
    data = pd.read_csv(
        'build/data_corrected.csv',
        skiprows=(1, 2, 3, 4, 5),
    )

    data = data.query('T > 255')

    func = lambda x, a, b: linear(x, a, b, x0=data['t'].mean())
    params, cov = curve_fit(func, data['t'], data['T'])
    heating_rate, _ = unc.correlated_values(params, cov)
    heating_rate *= u.kelvin / u.minute
    print('Rate = {}'.format(heating_rate))

    # fit activaiton energy
    T_max = data['T'][data['I_corrected'].argmax()] * u.kelvin
    print('T_max = {} '.format(T_max))

    data = data.drop_duplicates(subset=['T'])
    f = partial(diese_funktion_aus_der_anleitung, data=data, T_star=310)
    data['activation'] = data.apply(f, axis=1)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data['T_inv'] = 1 / data['T']
    # data = data.query('(T > 275)')

    fit_data = data.query('(0.00343 < T_inv < 0.00371)')
    ignored_data = data[~data.index.isin(fit_data.index)]
    func = partial(linear, x0=fit_data['T_inv'].mean())
    params, cov = curve_fit(
        func, fit_data['T_inv'], fit_data['activation'],
    )
    a, b = unc.correlated_values(params, cov)
    print(a, b)
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

    tau_0 = (
        relaxation_time /
        unp.exp(W.to(u.joule).magnitude / (const.k * T_max.magnitude))
    )
    print('Tau 0 : {}'.format(tau_0))

    with open('build/activation_work_2.tex', 'w') as f:
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
    plt.plot(
        ignored_data['T_inv'], ignored_data['activation'],
        '+', ms=4, color='#626262',
    )
    plt.plot(
        fit_data['T_inv'], func(fit_data['T_inv'], *params),
        color='darkgray',
    )
    plt.xlabel(r'$T^{-1} \mathrel{/} \si{\per\kelvin}$')
    plt.ylabel(r"$\ln{\frac{\int_T^{T'}i(T)\dif{T'}}{i(T)τ_0}}$")
    plt.tight_layout(pad=0)
    plt.savefig('build/method2.pdf')


    plt.figure()

    T = np.linspace(data['T'].min(), data['T'].max(), 1000)
    plt.plot(T, tau_0.to('s').magnitude.n * np.exp((W.to('J').magnitude.n / const.k / T)))
    plt.xlabel(r'$T \mathbin{/} \si{\kelvin}$')
    plt.ylabel(r'$\tau(T) \mathbin{/} \si{\second}$')
    plt.tight_layout(pad=1)
    plt.savefig('build/tau.pdf')


if __name__ == '__main__':
    main()
