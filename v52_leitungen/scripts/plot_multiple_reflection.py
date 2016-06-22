import matplotlib.pyplot as plt
import pandas as pd


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
    pulse['t'] -= nim['t'].loc[min_position]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot('t', 'U', data=pulse)
    ax2.plot('t', 'U', data=nim)

    ax2.set_xlabel(r'$t \mathbin{/} \si{\micro\second}$')
    ax2.set_ylabel(r'$U \mathbin{/} \si{\volt}$')
    ax1.set_ylabel(r'$U \mathbin{/} \si{\volt}$')

    ax2.set_xlim(-0.05, 0.6)
    ax2.set_ylim(-0.55, 0.15)

    # ax1.set_ylim(-0.62, 0.02)

    fig.tight_layout(pad=0, h_pad=0.3)
    fig.savefig('build/multiple_reflection.pdf')
