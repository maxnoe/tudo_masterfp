import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt
from uncertainties import correlated_values
# from IPython import embed

def curve(t, a_0, tau, u_0):
    return a_0*np.exp(-t/tau) + u_0

if __name__ == '__main__':
    print('Erster Anschluss')
    fig, (ax_1, ax_2) = plt.subplots(2, 1, sharex=True)
    df = pd.read_csv(
        './data/k1a6.csv',
        header=2, names=['t', 'U'],
    )
    # embed()
    df['t'] = (df.t - df.ix[df['U'].idxmax()].t)
    plateau = df.query('t > -0.8e-6 & t < -0.1e-6')
    u_1 = ufloat(plateau['U'].mean(), plateau['U'].std())
    ax_1.axhline(u_1.n, color='#ababab')
    ax_1.axhline(df['U'].max(), color='#ababab')
    reflection_factor = (df['U'].max() / u_1)

    R = - 50 * (1 + reflection_factor)/(1 - reflection_factor)
    print('resistance is {}'.format(R))

    df_fit = df.query('t > 0.4e-6 & t < 3e-6').dropna()
    popt, pcov = opt.curve_fit(curve, df_fit['t'], df_fit['U'], [1, 1e-6, 1])
    a_0, tau, u_0 = correlated_values(popt, pcov)
    print('tau is {}'.format(tau))
    L = tau * (50 + R)
    print('L is {}'.format(L))


    ax_1.plot(df['t']*1e6, df['U'], alpha=0.8, label='Gemessene Spannung')
    ax_1.plot(df_fit['t']*1e6, curve(df_fit['t'], a_0.n, tau.nominal_value, u_0.n), linewidth=2, label='Fit mit Exponentialfunktion')
    # ax_1.legend(loc='upper right')
    ax_1.set_ylabel(r'$U \mathbin{/} \si{\volt}$')
    ax_1.set_xlim(-2, 6)


    capacity_tex = '\SI{{{:.0f} \pm {:.0f}}}{{\\micro\\henry}}'.format(L.n * 1e6, L.s* 1e6)
    with open('build/k1a6_l.tex', 'w') as f:
        f.write(capacity_tex)

    r_tex = '\SI{{{:.0f} \pm {:.0f}}}{{\ohm}}'.format(R.n, R.s)
    with open('build/k1a6_r.tex', 'w') as f:
        f.write(r_tex)



    print('Zweiter Anschluss')
    df = pd.read_csv(
        './data/k2a2.csv',
        header=2, names=['t', 'U'],
    )
    df['t'] = df['t']
    local_min = df.query('t > -0.1e-6 & t < 2e-6')
    minimum = local_min.ix[local_min['U'].idxmin()]
    df['t'] = (df['t'] - minimum.t)
    # df['t'] = (df['t'] - minimum.t)
    plateau = df.query('t > -0.8e-6 & t < -0.2e-6')
    # plateau = df.query('t > 2e-6 & t < 3e-6')
    u_plateau = ufloat(plateau['U'].mean(), plateau['U'].std())
    reflection_factor = (minimum.U / u_plateau)
    ax_2.axhline(u_plateau.n, color='#ababab')
    ax_2.axhline(minimum.U, color='#ababab')
    # print(u_plateau, reflection_factor)

    R =  - 50 * (1 + reflection_factor)/(1 - reflection_factor)
    print('resistance is {}'.format(R))


    df_fit = df.query('t > 0.1e-6 & t < 1.5e-6').dropna()
    popt, pcov = opt.curve_fit(curve, df_fit['t'], df_fit['U'], [1, 1e-6, 1])
    a_0, tau, u_0 = correlated_values(popt, pcov)
    print('tau is {}'.format(tau))
    C = tau / (50 + R)
    print('C is {}'.format(C))

    ax_2.plot(df['t']*1e6, df['U'], alpha=0.8, label='Gemessene Spannung')
    ax_2.plot(df_fit['t']*1e6, curve(df_fit['t'], a_0.n, tau.nominal_value, u_0.n), linewidth=2, label='Fit mit Exponentialfunktion')

    fig.tight_layout(pad=0, h_pad=0.5)
    # print('Teh Z yo: {}'.format(usqrt(L/C)))
    ax_2.set_xlabel(r'$t \mathbin{/} \si{\micro\second}$')
    ax_2.set_ylabel(r'$U \mathbin{/} \si{\volt}$')
    fig.savefig('build/time_constant_fit.pdf')

    ind_tex = '\SI{{{:.2f} \pm {:.2f}}}{{\\nano\\farad}}'.format(C.n*1e9, C.s* 1e9)
    with open('build/k2a2_c.tex', 'w') as f:
        f.write(ind_tex)

    r_tex = '\SI{{{:.0f} \pm {:.0f}}}{{\ohm}}'.format(R.n, R.s)
    with open('build/k2a2_r.tex', 'w') as f:
        f.write(r_tex)


    #     f.write('  ' + key + ' & ')
    #     f.write('  {:3.1f} & '.format(delta_t.to('ns').magnitude))
    #     f.write('  {:2.2f} \\\\\n'.format(length.to('m').magnitude))
    #
    #     f.write(table_foot)
    # plt.show()
