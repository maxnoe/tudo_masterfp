import matplotlib.pyplot as plt
import pandas as pd
# import numpy as np
# from IPython import embed



def main():
    df = pd.read_csv('data/pulse_with_amp/F0001CH1.CSV', skipinitialspace=True, skiprows=18, header=None, usecols=[3,4], names=['time','voltage'])
    #guess correct units
    df.voltage /= 20
    df.time /= 5.000000E-07
    plt.plot(df.time, df.voltage)
    plt.xlabel('Zeit in Sekunden')
    plt.ylabel('Spannung in Volt')
    plt.tight_layout(pad=0.5)
    plt.savefig('build/plots/pulse_with_amp.pdf')



if __name__ == '__main__':
    main()
