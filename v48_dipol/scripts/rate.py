import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.constants as const
from pint import UnitRegistry
u = UnitRegistry()

import uncertainties as unc
from IPython import embed


def linear(x, a, b, x0):
    return a * (x - x0) + b

def main():
    data = pd.read_csv(
        'data/messwerte_2.csv',
        skiprows=(1, 2, 3, 4, 5),
    )
    data['T'] = data['T'].apply(const.C2K)

    func = lambda x, a, b: linear(x, a, b, x0=data['t'].mean())
    params, cov = curve_fit(func, data['t'], data['T'])
    a, b = unc.correlated_values(params, cov)
    print('Rate = {} Kelvin per minute'.format(a))
    # plt.plot(data['t'], data['T'], '+', ms=2)
    # plt.plot(data['t'], func(data['t'], *params), label='Ausgleichsgerade', color='gray')

    T_max = 289.95 * u.kelvin
    # embed()
    relaxation_time =(u.boltzmann_constant *  T_max**2)/(0.526 * u.eV *a * u.kelvin/u.minute)  * np.exp(- 0.526 *  u.eV/(u.boltzmann_constant *  T_max))
    print(relaxation_time.to(u.second))
    # embed()
    # plt.tight_layout(pad=0)
    # plt.show()
    # plt.savefig('build/rate.pdf')


if __name__ == '__main__':
    main()
