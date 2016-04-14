import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.constants as c
from pint import UnitRegistry
u = UnitRegistry()


def scattering(theta, z=2, Z=79, alpha_energy=5.408 * u.MeV):
    theta = theta.to('radian')
    angle_dependency = 1 / (np.sin(theta/2)**4)
    a = 1 / ((4 * np.pi * epsilon_0)**2)
    b = (z * Z * e**2 / (4 * alpha_energy))**2
    return a * b * angle_dependency


df = pd.read_csv('data/scattering_gold_2_micrometer.csv', comment='#', index_col=0)


top = r'''\begin{tabular}{S[table-format=1.0] S[table-format=3.0]}
    \toprule
    {$\theta \mathbin{/} \si{\degree}$} & {$N$} \\
    \midrule
'''
bottom = r'''    \bottomrule
\end{tabular}
'''
with open('build/scattering_gold_2_micrometer.tex', 'w') as f:

    f.write(top)

    temp = r'    {} & {} \\'
    for row in df.itertuples():
        f.write(temp.format(row.Index, row.counts))
        f.write('\n')
    f.write(bottom)


dx = 2 * u.mm
dy = 10 * u.mm
l = (41 + 4) * u.mm
delta_omega = np.arctan(dy / (2 * l)) * np.arctan(dx / (2 * l)) * 4
print('delta_omega is {}'.format(delta_omega))

density_gold = 19.3 * (u.kg / u.m**3)
thickness = 2 * u.micrometer
gold_mass = 197 * u.gram / u.mol
n_0 = (density_gold * thickness * c.N_A / u.mol) / gold_mass
print('n_0 is {}'.format(n_0.to('1/cm^2')))

R = (41 + 39 + 17 + 4) * u.mm
A_detector = 2 * u.mm * 10 * u.mm
N_source = (A_detector / (4 * np.pi * R**2)) * (318000) / u.second
print('N_source is {} '.format(N_source.to('1/s')))

N_meassurement = (df.counts.values / 60)/u.second
print('N_meassurement is {}'.format(N_meassurement))

cross_section = N_meassurement / (N_source * n_0 * delta_omega)
# print((N_source * n_0 * delta_omega).to('1/cm^2  1/s  1/radian^2'))
print('cross_section is {}'.format(cross_section.to('barn/degree')))

plt.plot(
    df.index,
    cross_section.to('barn/radian'),
    '+',
    label='Gemessener Streuquerschnitt',
)

e = (c.e * u.coulomb)
epsilon_0 = c.epsilon_0 * (u.ampere * u.second / (u.volt * u.meter))
ts = np.linspace(0, df.index.max(), 2000) * u.degree
alpha_energy = 5.408 * u.MeV
scatter = scattering(
    theta=ts,
    z=2,
    Z=79,
    alpha_energy=alpha_energy,
).to('barn/radian')

plt.plot(
    ts[scatter.magnitude < 1e9],
    scatter[scatter.magnitude < 1e9],
    '-',
    label='Theoretischer Streuquerschnitt',
)
plt.xlabel('Winkel in Grad')
plt.ylabel(r'$\dd{\sigma}{\Omega} \Biggm/ \si{\barn\per\steradian}$')

plt.legend(loc='best')
plt.tight_layout(pad=0.4)
plt.savefig('build/plots/cross_section.pdf')
