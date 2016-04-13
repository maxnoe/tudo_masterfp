import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.constants as c
from pint import UnitRegistry
u = UnitRegistry()

def scattering(theta , z = 2, Z = 79, alpha_energy = 5.408 * u.MeV):
    theta = theta.to('radian')
    angle_dependency = 1 / ( np.sin(theta/2)**4)
    a = 1 / ((4 * np.pi * epsilon_0 )**2)
    b = (z * Z * e**2 / (4 * alpha_energy) )**2
    return a * b * angle_dependency


df  = pd.read_csv('data/scattering_gold_2_micrometer.csv', comment='#', index_col=0)

dx = 2 *u.mm
dy = 10 * u.mm
l = (41 + 4) *  u.mm
delta_omega = np.arctan(dy/(2*l)) * np.arctan(dx/(2*l))*4
print('delta_omega is {}'.format(delta_omega))

density_gold = 19.3 * (u.kg / u.m**3 )
thickness = 2 * u.micrometer
gold_mass = 197 *u.gram/u.mol
n_0 = (density_gold * thickness * c.N_A / u.mol) / gold_mass
print('n_0 is {}'.format(n_0.to('1/cm^2')))

R =  (41 + 39 + 17 + 4) * u.mm
A_detector = 2*np.pi * (0.5 * u.cm)**2
N_source = (A_detector / (4 * np.pi * R**2 )) * (318000)/u.second
print('N_source is {} '.format(N_source.to('1/s')))

N_meassurement = (df.counts.values / 60)/u.second
print('N_meassurement is {}'.format(N_meassurement))

cross_section = N_meassurement / (N_source * n_0 * delta_omega)
# print((N_source * n_0 * delta_omega).to('1/cm^2  1/s  1/radian^2'))
print('cross_section is {}'.format(cross_section.to('barn/degree')))

plt.plot(df.index, cross_section.to('barn/radian^2'), '+', label='Gemessener Streuquerschnitt')

e = (c.e * u.coulomb)
epsilon_0 = c.epsilon_0 * (u.ampere * u.second / (u.volt * u.meter))
ts = np.linspace(0, df.index.max(), 2000) * u.degree
alpha_energy = 5.408 * u.MeV
scatter =  scattering(theta = ts, z = 2, Z =  79, alpha_energy =  alpha_energy).to('barn/degree')

plt.plot(ts[scatter.magnitude < 1e9], scatter[scatter.magnitude < 1e9], '-', label='Theoretischer Streuquerschnitt')
plt.xlabel('Winkel in Grad')
plt.ylabel(r'$\dd{\sigma}{\Omega} \Biggm/ \si{\barn\per\degree}$')

plt.legend()
# plt.show()



# df  = pd.read_csv('data/scattering_gold_4_micrometer.csv', comment='#', index_col=0)
# plt.plot(df.index, df.counts, '+', label='4 ')
# plt.tight_layout()
plt.savefig('build/plots/cross_section.pdf')
