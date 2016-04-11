import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as c
from pint import UnitRegistry
from joblib import Parallel, delayed
from scipy.integrate import quad

u = UnitRegistry()
Q = u.Quantity

R_air = 287.058 * u.joule / (u.kilogram * u.kelvin)  # dry air

m_e = c.m_e * u.kg
epsilon_0 = c.epsilon_0 * (u.ampere * u.second / (u.volt * u.meter))

I = 14 * u.eV

pressures = range(20, 1001, 10)


def gas_density(p, T, R_specific):
    return p / (R_specific * T.to('kelvin'))


def electron_density(rho):
    # assume air is made of nitrogen and nothing else. -> We all die.
    Z = 7
    A = Z + 7  # plus 7 neutrons
    n = (Z * rho) / (A * u.amu)  # multiply atomic mass unit
    return n


def inv_bethe(E, n, z, I, m):
    E = E * u.MeV
    v = np.sqrt(2 * E / m)
    ln = np.log((2 * m_e * v**2) / I)
    a = (4 * np.pi * n * z**2) / (m_e * v**2) * \
        ((c.e * u.coulomb)**2 / (4 * np.pi * epsilon_0))**2
    return 1 / (a * ln).to('MeV / cm').magnitude


def mean_free_path(p):
    rho = gas_density(p*u.mbar, Q(20, u.celsius), R_air)

    alpha_energy = 5.408 * u.MeV
    alpha_mass = c.physical_constants['alpha particle mass'][0] * u.kg

    result, err = quad(
        inv_bethe,
        alpha_energy.magnitude,
        0,
        args=(electron_density(rho), 2, I, alpha_mass)
    )

    return result


if __name__ == '__main__':
    plt.style.use('ggplot')

    with Parallel(4, verbose=10) as pool:
        distances = np.array(pool(delayed(mean_free_path)(p) for p in pressures))

    plt.xlabel('pressure in mb')
    plt.ylabel('mean free path in cm')
    plt.xlim(0, 1000)
    plt.plot(pressures, -distances, '+', label='10000')
    plt.savefig('build/plots/mean_free_path.pdf')
