import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as c
from pint import UnitRegistry

u = UnitRegistry()
Q = u.Quantity

R_air = 287.058 * u.joule / (u.kilogram * u.kelvin)  # dry air

m_e = c.m_e * u.kg
epsilon_0 = c.epsilon_0 * (u.ampere * u.second / (u.volt * u.meter))


def gas_density(p, T, R_specific):
    return p / (R_specific * T.to('kelvin'))


def electron_density(rho):
    # assume air is made of nitrogen and nothing else. -> We all die.
    Z = 7
    A = Z + 7  # plus 7 neutrons
    n = (Z * rho) / (A * u.amu)  # multiply atomic mass unit
    return n


def bethe(n, z, v, I):
    ln = np.log((2 * m_e * v**2) / I)
    a = (4 * np.pi * n * z**2) / (c.m_e * u.kilogram * v**2) * \
        ((c.e * u.coulomb)**2 / (4 * np.pi * epsilon_0))**2
    return a * ln


alpha_energy = 5.408 * u.MeV
alpha_mass = c.physical_constants['alpha particle mass'][0] * u.kg
speed = np.sqrt(2 * alpha_energy / alpha_mass)

# nitrogen exitation?
I = 14 * u.eV

pressure = np.linspace(0, 250, 1000) * u.mbar
rho = gas_density(pressure, Q(20, u.celsius), R_air)
stopping = bethe(electron_density(rho), 4, speed, I).to('MeV/cm')


plt.plot(pressure.to('millibar'), stopping)
plt.xlabel(r'$p \mathrel{/} \si{\milli\bar}$')
plt.ylabel(r'$\dd{E}{x} \Biggm/ \si{\mega\electronvolt\per\centi\meter}$')
plt.tight_layout(pad=0)
plt.savefig('build/plots/bethe_air.pdf')
