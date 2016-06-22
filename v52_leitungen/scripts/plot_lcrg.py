import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    data = pd.read_csv('./data/rg058-85m-lcrg.csv', header=1)

    data['G'] = data['R'] * data['C'] * 1e-9 / (data['L'] * 1e-6) * 1e3

    fig, axs = plt.subplots(4, 1, sharex=True)

    quantities = ('R', 'L', 'C', 'G')
    units = (r'\ohm', r'\micro\henry', r'\nano\farad', r'\milli\siemens')
    for ax, key, unit in zip(axs, quantities, units):

        ax.plot('f', key, 'o', data=data, ms=4, mew=0)


        ax.set_ylabel(r'${} \mathbin{{/}} \si{{{}}} $'.format(key, unit))
        ax.set_xlabel('')

    axs[0].set_ylim(2.3, 3.4)

    axs[1].set_ylim(20.0, 21.3)
    axs[1].set_yticks(np.arange(20.0, 21.3, 0.3))

    axs[2].set_yticks(np.arange(8.5, 8.61, 0.02))

    axs[3].set_xlabel(r'$f \mathbin{/} \si{\hertz}$')

    axs[3].set_xscale('log')
    axs[3].set_xlim(90, 1.1e5)
    axs[3].set_ylim(0.9, 1.45)

    fig.tight_layout(pad=0, h_pad=0.5)
    fig.savefig('build/lcrg.pdf')
