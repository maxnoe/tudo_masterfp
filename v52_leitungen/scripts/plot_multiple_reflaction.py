import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    pulse = pd.read_csv('./data/mehrfach_reflektion.csv', header=2, names=['t', 'U'])
    nim = pd.read_csv('./data/mehrfach_reflektion_nim.csv', header=2, names=['t', 'U'])

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot('t', 'U', data=pulse)
    ax2.plot('t', 'U', data=nim)
    plt.show()
