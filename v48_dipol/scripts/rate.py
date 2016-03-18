import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.constants as const
from pint import UnitRegistry
import uncertainties as unc
from functools import partial

u = UnitRegistry()


def linear(x, a, b, x0):
    return a * (x - x0) + b


def main():
    data = pd.read_csv(
        'data/messwerte_2.csv',
        skiprows=(1, 2, 3, 4, 5),
    )
    data['T'] = data['T'].apply(const.C2K)

    func = partial(linear, x0=data['t'].mean())

    params, cov = curve_fit(func, data['t'], data['T'])
    a, b = unc.correlated_values(params, cov)
    print('Rate = {} Kelvin per minute'.format(a))

    with open('build/rate.tex', 'w') as f:
        f.write(r'b = \SI{{{a.n:1.2f} +- {a.s:1.2f}}}{{\kelvin\per\minute}}'.format(a=a))

    T_max = 289.95 * u.kelvin
    relaxation_time = (
        (u.boltzmann_constant * T_max**2) /
        (0.526 * u.eV * a * u.kelvin / u.minute) *
        np.exp(-0.526 * u.eV / (u.boltzmann_constant * T_max))
    )
    print(relaxation_time.to(u.second))

    t = np.linspace(0, 60, 2)
    plt.plot(t, func(t, *params), label='Ausgleichsgerade', color='gray')
    plt.plot(data['t'], data['T'], 'x', ms=3, label='Messwerte')
    plt.xlabel(r'$t \mathbin{/} \si{\minute}$')
    plt.ylabel(r'$T \mathbin{/} \si{\kelvin}$')
    plt.legend(loc='lower right')

    plt.tight_layout(pad=0)
    plt.savefig('build/rate.pdf')


if __name__ == '__main__':
    main()
