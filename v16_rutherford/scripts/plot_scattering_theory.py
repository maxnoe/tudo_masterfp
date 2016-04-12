import matplotlib
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import scipy.constants as c
from pint import UnitRegistry
u = UnitRegistry()

e = (c.e * u.coulomb)
epsilon_0 = c.epsilon_0 * (u.ampere * u.second / (u.volt * u.meter))

def scattering(theta , z = 2, Z = 79, alpha_energy = 5.408 * u.MeV):
    theta = theta.to('radian')
    angle_dependency = 1 / ( np.sin(theta/2)**4)
    a = 1 / ((4 * np.pi * epsilon_0 )**2)
    b = (z * Z * e**2 / (4 * alpha_energy) )**2
    return a * b * angle_dependency




ts = np.linspace(0, 0.2, 200) * u.degree
alpha_energy = 5.408 * u.MeV

# alpha_mass = c.physical_constants['alpha particle mass'][0] * u.kg

scatter =  scattering(theta = ts, z = 2, Z =  79, alpha_energy =  alpha_energy).to('barn/radian')
# print(scatter[20])
plt.plot(ts, scatter, '+')
plt.ylim(0, 1e9)
plt.xlabel('Winkel in Grad')
plt.ylabel('Anzahl')

# df  = pd.read_csv('data/scattering_gold_4_micrometer.csv', comment='#', index_col=0)
# plt.plot(df.index, df.counts, '+', label='4 ')
plt.tight_layout()
plt.show()
# plt.savefig('build/plots/scattering_theory.pdf')
