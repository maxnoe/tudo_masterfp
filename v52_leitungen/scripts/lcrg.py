import numpy as np
import pint
from scipy import constants

u = pint.UnitRegistry()


def r(rs, d, D):
    r = rs / (2 * np.pi) * (1 / d + 1 / D)
    return r


def l(mu, d, D):
    l = mu/(2*np.pi) * np.log(D/d)
    return l


def g(sigma, d, D):
    g = 2 * np.pi * sigma/(np.log(D/d))
    return g


def c(epsilon, d, D):
    c = 2 * np.pi * epsilon/(np.log(D/d))
    return c


def main():
    d = 0.9 * u.mm
    D = 2.95 * u.mm
    epsilon = constants.epsilon_0 * u.ampere * u.second / (u.volt * u.meter)
    epsilon_r = 2.25
    mu = constants.mu_0 * u.newton / u.ampere**2
    c_koax = c(epsilon_r*epsilon, d, D)

    l_koax = l(mu, d, D)
    print('Kapazitätsbelag: {}, Induktivitätsbelag: {}'.format(c_koax.to('nF / m'), l_koax.to('nH / m')))
    # mu_r = 1
    # Rs = np.sqrt(np.pi )

    v_phase = 1 / np.sqrt(l_koax * c_koax)
    print('Phasengeschwindigkeit: {}'.format(v_phase.to('km/s')))

if __name__ == '__main__':
    main()
