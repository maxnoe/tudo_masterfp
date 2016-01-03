import matplotlib.style
matplotlib.style.use('ggplot')
# matplotlib.style.use('matplotlibrc')
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as c
from pint import UnitRegistry
u = UnitRegistry()
u.define('electronvolts = 10E-19 * joule = eV')
u.define('megaelectronvolts = 10E-13 * joule = MeV')

from IPython import embed
def electron_density(rho):
    ## assume air is made of nitrogen and nothing else. -> We all die.
    Z = 7
    A = Z + 7 #plus 7 neutrons
    n = ( Z *rho ) / ( A *u.amu ) # multiply atomic mass unit
    return n

def bethe(n, z, v, I):
    ln = np.log((2 * c.m_e*u.kilogram * v**2) / I)
    eps = c.epsilon_0*(u.ampere*u.second/(u.volt*u.meter))
    a = (4 * np.pi * n * z**2)/(c.m_e*u.kilogram * v**2) * ((c.e*u.coulomb)**2 / (4 *np.pi *eps))**2
    return a*ln


am_energy = 5.408 * u.MeV
alpha_mass = 4*u.amu
speed = np.sqrt(2*am_energy /  alpha_mass).to('meter/second') #meter per second

#nitrogen exitation?
I = 14 * u.eV

rhos = np.linspace(0, 1, 1000)*(u.kg/(u.meter**3))
stopping = bethe(electron_density(rhos), 4, speed, I).to('MeV/m')

r_specific = 287.058 *u.joule /(u.kilogram*u.kelvin)#dry air
pressure = rhos * r_specific * 20*u.kelvin # assume room temp

# print(pressure.to('millibar'))
plt.plot(pressure.to('millibar'), stopping)
plt.xlabel("Druck in millibar")
plt.ylabel("Energieverlus pro Strecke in MeV/m")
plt.axhline(y=am_energy.magnitude, linestyle='--', color='darkgray', label="Mittlere Energie der Alpha-Teilchen")
plt.legend()
plt.savefig("bethe.pdf")
