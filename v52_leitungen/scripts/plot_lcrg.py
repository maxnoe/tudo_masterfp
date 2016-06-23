import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml


if __name__ == '__main__':
    with open('build/length.yaml') as f:
        length = yaml.safe_load(f)['85m']['length']

    data = pd.read_csv('./data/rg058-85m-lcrg.csv', header=1)
    for key in ('R', 'C', 'L'):
        data[key] = data[key] / length

    data['G'] = data['R'] * data['C'] * 1e-9 / (data['L'] * 1e-6) * 1e6

    fig, axs = plt.subplots(4, 1, sharex=True)

    quantities = ('R', 'L', 'C', 'G')
    units = (r'\ohm', r'\micro\henry', r'\nano\farad', r'\micro\siemens')
    for ax, key, unit in zip(axs, quantities, units):

        ax.plot('f', key, 'o', data=data, ms=4, mew=0)

        ax.set_ylabel(r'${} \mathbin{{/}} \si[per-mode=reciprocal]{{{}\per\meter}}$'.format(key, unit))
        ax.set_xlabel('')

    axs[0].set_yticks(np.arange(0.0275, 0.041, 0.0025))

    axs[1].set_ylim(0.235, 0.255)
    axs[1].set_yticks(np.arange(0.235, 0.2551, 0.005))

    axs[2].set_yticks(np.arange(0.1, 0.10125, 0.00025))

    axs[3].set_xlabel(r'$f \mathbin{/} \si{\hertz}$')

    axs[3].set_xscale('log')
    axs[3].set_xlim(90, 1.1e5)
    # axs[3].set_ylim(0.9, 1.5)
    axs[3].set_yticks(np.arange(11, 18, 2))

    fig.tight_layout(pad=0, h_pad=0.5)
    fig.savefig('build/lcrg.pdf')
