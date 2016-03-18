import pandas as pd
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


if __name__ == '__main__':
    data = pd.read_csv(
        'data/messwerte_2.csv',
        skiprows=(1, 2, 3, 4, 5),
    )
    data['T'] = data['T'].apply(const.C2K)
    fit_start = data['T'][data['I'].argmin()]
    fit1 = data.query('({} <= T <= 270) | (T > 310)'.format(fit_start)).iloc[:-1]

    T0 = fit1['T'][data['I'].argmin()]
    func = exponential
    params = 10, 0.1, data['I'].min()
    params, cov = curve_fit(
        func, fit1['T'] - T0, fit1['I'],  params
    )
    data['I_corrected'] = data['I'] - func(data['T'] - T0, *params)
    data.to_csv('build/data_corrected.csv', index=False)

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
    plt.savefig('build/data_correction_fit.pdf')

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
    plt.savefig('build/data_corrected.pdf')
