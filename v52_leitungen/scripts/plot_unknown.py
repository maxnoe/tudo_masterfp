import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fig, axs = plt.subplots(3, 1, sharex=True)

    for ax, label in zip(axs, ('k1a4', 'k2a2', 'k1a6')):

        df = pd.read_csv(
            './data/{}.csv'.format(label),
            header=2, names=['t', 'U'],
        )

        df['t'] = (df['t'] - df['t'].min()) * 1e6
        ax.plot('t', 'U', data=df, label=label)
        ax.set_ylabel(r'$U \mathbin{/} \si{\volt}$')
        ax.legend(loc='upper left')

    axs[2].set_xlabel(r'$t \mathbin{/} \si{\micro\second}$')
    axs[2].set_xlim(0, 5)

    fig.tight_layout(pad=0, h_pad=0.5)
    fig.savefig('build/unknown.pdf')
