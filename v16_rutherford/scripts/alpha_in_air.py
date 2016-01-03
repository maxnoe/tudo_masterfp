import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

alpha_energy = 5486e3 * const.e  # J
alpha_mass = const.physical_constants['alpha particle mass'][0]
R_air = 287.058  # J / (kg K)
air_percentages = {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01}
molar_mass = {'N2': 28e-3, 'O2': 32e-3, 'Ar': 40e-3}  # kg / mol
Z = {'N2': 14, 'O2': 16, 'Ar': 18}
I_J_per_mol = {'N2': 1402e3, 'O2': 1313.9e3, 'Ar': 1520.8e3}  # J / Mol


molar_mass_air = sum(air_percentages[k] * molar_mass[k] for k in molar_mass)
Z_air = sum(air_percentages[k] * Z[k] for k in molar_mass)
I_air = sum(air_percentages[k] * I_J_per_mol[k] / const.N_A for k in molar_mass)
Z_alpha = 2


def air_density(p, T):
    return p / (R_air * T)


def bethe(z, Z, N, v, I):
    fac = const.e**4 / (4 * const.pi * const.epsilon_0 ** 2)
    ln = np.log(2 * const.m_e * v**2 / I)
    return fac * z**2 * N * Z / (const.m_e * v**2) * ln


def main():
    T = const.C2F(20)
    p = np.logspace(-3, 5, 100)  # Pa
    N = air_density(p, T) / molar_mass_air * const.N_A

    v = np.sqrt(2 * alpha_energy / alpha_mass)
    loss = bethe(Z_alpha, Z_air, N, v, I_air)

    plt.plot(p, loss)
    plt.ylabel(
        r'$\frac{\mathrm{d}E}{\mathrm{d}x} \,/\, \frac{\mathrm{J}}{\mathrm{m}}$'
    )
    plt.xlabel(r'$p \,/\, \mathrm{Pa}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()
