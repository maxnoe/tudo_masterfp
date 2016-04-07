import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as c
from pint import UnitRegistry
from tqdm import tqdm

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



I = 14 * u.eV

mean_free_paths = []
pressures = range(20, 1001, 10)
for p in tqdm(pressures):
    rho = gas_density(p*u.mbar, Q(20, u.celsius), R_air)
    distances, step = np.linspace(0, 40, 1000, retstep=True)

    alpha_energy = 5.408 * u.MeV
    alpha_mass = c.physical_constants['alpha particle mass'][0] * u.kg

    #calculate the mean free path by iterativaly adapting the alpha energy
    for d in distances:
        delta_x = step*u.cm
        speed = np.sqrt(2 * alpha_energy / alpha_mass)
        energy_loss = (bethe(electron_density(rho), 4, speed, I) * delta_x).to('MeV')
        if energy_loss < alpha_energy:
                alpha_energy = alpha_energy - energy_loss
                # energies.append(alpha_energy)
                # print('Distance: {}, Loss: {} , Energy: {}'.format(d, energy_loss, alpha_energy))
        else:
            # print('Mean free path for {} mbar is: {}'.format(p, d*u.cm))
            mean_free_paths.append(d)
            break

plt.xlabel('pressure in mb')
plt.ylabel('mean free path in cm')
plt.plot(pressures, mean_free_paths, 'b+')
plt.savefig('build/plots/mean_free_path.pdf')
