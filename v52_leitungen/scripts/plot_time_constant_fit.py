import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from uncertainties import ufloat
# from IPython import embed

def curve(t, a_0, tau, u_0):
    return a_0*np.exp(-t*tau) + u_0

if __name__ == '__main__':
    fig, ax = plt.subplots()
    df = pd.read_csv(
        './data/k1a6.csv',
        header=2, names=['t', 'U'],
    )
    # embed()
    df['t'] = (df.t - df.ix[df['U'].idxmax()].t) * 1e6
    ax.plot('t', 'U', data=df, label='Kasten 1 Abschluss 6', alpha=0.8)
    reflection_factor = (df['U'].max() / df.query('t > -0.6 & t < -0.1')['U'].mean())
    ax.hline(df['U'].max())
    ax.hline(df.query('t > -0.6 & t < -0.1')['U'].mean())

    R = - 50 * (1 + reflection_factor)/(1 - reflection_factor)
    print('resistance is {}'.format(R))

    df_fit = df.query('t > 0.4 & t < 3').dropna()
    # ax.set_ylabel(r'$U \mathbin{/} \si{\volt}$')
    ax.legend(loc='upper left')
    (a_0, tau, u_0), pcov = opt.curve_fit(curve, df_fit['t'], df_fit['U'])
    print('tau is {}'.format(tau))
    ax.plot(df_fit['t'], curve(df_fit['t'], a_0, tau, u_0))
    # ax.set_xlabel(r'$t \mathbin{/} \si{\micro\second}$')
    ax.set_xlim(-2, 6)
    tau = ufloat(tau, np.sqrt(pcov[1,1]))
    L = tau / (50 + R)
    print('L is {}'.format(L))

    fig.tight_layout(pad=0, h_pad=0.5)
    fig.savefig('build/time_constant_fit.pdf')
    # plt.show()
