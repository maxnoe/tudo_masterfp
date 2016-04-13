import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as c
from pint import UnitRegistry

u = UnitRegistry()
Q = u.Quantity

R_air = 287.058 * u.joule / (u.kilogram * u.kelvin)  # dry air

m_e = c.m_e * u.kg
epsilon_0 = c.epsilon_0 * (u.ampere * u.second / (u.volt * u.meter))


def gas_density(p, T = Q(20, u.celsius), R_specific = 287.058 * u.joule / (u.kilogram * u.kelvin)  ):
    return p / (R_specific * T.to('kelvin'))


def electron_density(rho, Z = 7, A = 14):
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
I = 82 * u.eV

pressure = np.linspace(0, 1100, 2000) * u.mbar
rho = gas_density(pressure, Q(20, u.celsius), R_air)
stopping = bethe(electron_density(rho), 4, speed, I).to('MeV/cm')


plt.plot(pressure.to('millibar'), stopping)
plt.xlabel(r'$p \mathrel{/} \si{\milli\bar}$')
plt.ylabel(r'$\dd{E}{x} \Biggm/ \si{\mega\electronvolt\per\centi\meter}$')
#draw energy of alpha particles form am 241
am_energy = 5.408 * u.MeV
# plt.axhline(y=am_energy.magnitude, linestyle='--', color='darkgray', label="Mittlere Energie der Alpha-Teilchen")
# plt.legend(loc='lower right', fancybox=True, framealpha=0.4)
plt.tight_layout(pad=0)
plt.savefig('build/plots/bethe_air.pdf')
