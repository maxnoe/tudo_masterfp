import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import fftpack


if __name__ == '__main__':

    rectangle_short = pd.read_csv(
        './data/att_short_102_9kHz_fft.csv',
        header=2, names=['f', 'A']
    )
    rectangle_long = pd.read_csv(
        './data/att_long_102_9kHz_fft.csv',
        header=2, names=['f', 'A']
    )

    fig, ax = plt.subplots()
    ax.plot('f', 'A', data=rectangle_long, label='85m-Kabel')
    ax.plot('f', 'A', data=rectangle_short, label='kurzes Kabel')
    ax.legend()
    plt.show()
