import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    data = pd.read_csv('./data/rg058-85m-lcrg.csv', header=1)

    fig, axs = plt.subplots(3, 1)

    for ax, key in zip(axs, ('R', 'L', 'C')):
        ax.plot('f', key, 'o', data=data)
        ax.set_title(key)
        ax.set_xscale('log')
        ax.set_xlabel('f / Hz')
        ax.set_xlim(90, 1.1e5)

    plt.show()
