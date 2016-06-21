import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    df = pd.read_csv(
        'data/reflek1.csv',
        skiprows=(0, 1),
        names=['t', 'U'],
        header=None,
    )
    df['t'] *= 1e9

    fig, ax = plt.subplots()

    df.plot('t', 'U', ax=ax, legend=False)

    ax.set_xlabel(r'$t \mathbin{/} \si{\nano\second}$')
    ax.set_ylabel(r'$U \mathbin{/} \si{\volt}$')

    fig.tight_layout(pad=0)
    fig.savefig('build/length_measurement.pdf')
