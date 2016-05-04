import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    df1 = pd.read_csv('./data/k1a4.csv', header=2, names=['t', 'U'])
    df2 = pd.read_csv('./data/k2a2.csv', header=2, names=['t', 'U'])
    df3 = pd.read_csv('./data/k1a6.csv', header=2, names=['t', 'U'])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot('t', 'U', data=df1)
    ax2.plot('t', 'U', data=df2)
    ax3.plot('t', 'U', data=df3)

    plt.show()
