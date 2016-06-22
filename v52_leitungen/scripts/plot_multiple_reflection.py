import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

color = [e['color'] for e in plt.rcParams['axes.prop_cycle']]

T = 0.051
d = 0.0025
linestyle = (0, (3, 1))


def a(*elems):
    return np.array(elems)


if __name__ == '__main__':

    pulse = pd.read_csv(
        './data/mehrfach_reflektion.csv',
        header=2, names=['t', 'U'],
    )
    nim = pd.read_csv(
        './data/mehrfach_reflektion_nim.csv',
        header=2, names=['t', 'U'],
    )

    pulse['t'] *= 1e6
    nim['t'] *= 1e6

    pulse['U'] -= pulse['U'].loc[:300].mean()

    max_position = pulse['U'].argmax()
    pulse['t'] -= pulse['t'].loc[max_position]
    min_position = nim['U'].argmin()
    nim['t'] -= nim['t'].loc[min_position]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot('t', 'U', data=pulse)
    ax1.set_ylabel(r'$U \mathbin{/} \si{\volt}$')

    ax2.plot('t', 'U', data=nim)
    ax2.set_ylabel(r'$U \mathbin{/} \si{\volt}$')
    ax2.set_ylim(-0.6, 0.2)

    ax3.axhline(10, color='black')
    ax3.axhline(20, color='black')

    ax3.set_yticks([0, 10, 20])
    ax3.set_ylabel(r'$l \mathbin{/} \si{\meter}$')
    ax3.set_xlabel(r'$t \mathbin{/} T$')
    ax3.set_xticks(T * np.arange(7))

    ax3.plot(T * a(0, 2, 4, 6), a(0, 20, 0, 20), color=color[0])
    ax3.plot(T * a(1, 2, 4, 6), a(10, 0, 20, 0), color=color[1])
    ax3.plot(T * a(3, 4, 6) + d, a(10, 0, 20), color=color[1])
    ax3.plot(T * a(5, 6) + d, a(10, 0), linestyle=linestyle)
    ax3.plot(T * a(5, 6) + 2 * d, a(10, 0), color=color[1], linestyle=linestyle)

    ax3.set_xlabel(r'$t \mathbin{/} \si{\micro\second}$')
    ax3.set_xlim(-0.05, 0.35)

    fig.tight_layout(pad=0, h_pad=0.3)
    fig.savefig('build/multiple_reflection.pdf')
