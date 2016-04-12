import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp
import scipy.constants as c

from pint import UnitRegistry
u = UnitRegistry()
Q = u.Quantity

m_e = c.m_e * u.kg
epsilon_0 = c.epsilon_0 * (u.ampere * u.second / (u.volt * u.meter))
alpha_energy = 5.408 * u.MeV
alpha_mass = c.physical_constants['alpha particle mass'][0] * u.kg
x0 = 10.1 * u.cm


def linear(x, m, b):
    return m * x + b


def find_maxima(folder):
    data = pd.read_csv(
        path.join('data', folder, 'pressures.csv'),
        comment='#',
    )

    template = path.join('data', folder, 'TEK{:04d}.CSV')

    maxima = np.zeros_like(data.index)

    for i, row in enumerate(data.itertuples()):
        pulse = pd.read_csv(
            template.format(row.filenum),
            skiprows=18,
            header=None,
            usecols=(3, 4),
            names=['t', 'U'],
        )

        pulse['smooth'] = pulse['U'].rolling(50, center=True).mean()
        maxima[i] = pulse['smooth'].max()

    data['max'] = maxima
    return data


def gas_density(p, T = (273.15 + 20) * u.kelvin, R_specific = 287.058 * u.joule / (u.kilogram * u.kelvin)  ):
    return p / (R_specific * T)


def electron_density(rho, Z=7, A=14):
    # assume air is made of nitrogen and nothing else. -> We all die.
    n = (Z * rho) / (A * u.amu)  # multiply atomic mass unit
    return n


def bethe(E, n, z, I, m):
    print(E, n, z, I, m)
    v = unp.sqrt((2 * E / m).to_base_units().magnitude) * (u.meter/u.second)
    print(v.to('m/s'))

    ln = unp.log(((2 * m_e * v**2) / I).to_base_units().magnitude)

    a = (4 * np.pi * n * z**2) / (m_e * v**2)
    b = ((c.e * u.coulomb)**2 / (4 * np.pi * epsilon_0))**2

    return  (a * b * ln).to('MeV / cm')


def main():

    with_foil = find_maxima('with_foil')
    without_foil = find_maxima('without_foil')

    params_with, cov_with = curve_fit(linear, with_foil['p'], with_foil['max'])
    params_without, cov_without = curve_fit(
        linear, without_foil['p'], without_foil['max']
    )

    m_w, b_w = unc.correlated_values(params_with, cov_with)
    m_wo, b_wo = unc.correlated_values(params_without, cov_without)

    rho_gold = 19.32 * u.gram / (u.cm**3)
    density_gold = electron_density(rho_gold, Z = 79, A = 197)


    delta_E = alpha_energy*(1 - b_w/b_wo)
    print(delta_E)
    mean_E = alpha_energy - 0.5*delta_E

    print(mean_E)
    thickness =  delta_E /  bethe(E = mean_E, n = density_gold, m = alpha_mass, I = 790*u.eV , z= 2)
    print(thickness.to('micrometer'))

    p = np.linspace(0, 350, 2)

    plt.plot('p', 'max', '+', data=with_foil, label='Mit Folie')
    plt.plot(p, linear(p, *params_with))
    plt.plot('p', 'max', '+', data=without_foil, label='Ohne Folie')
    plt.plot(p, linear(p, *params_without))
    plt.xlabel(r'$p \mathrel{/} \si{\milli\bar}$')
    plt.ylabel(r'$\bar{U}_\text{max} \mathrel{/} \si{\volt}$')
    plt.legend()
    plt.ylim(0, 70)
    plt.tight_layout(pad=0)
    plt.savefig('build/plots/gold_thickness.pdf')


if __name__ == '__main__':
    main()
