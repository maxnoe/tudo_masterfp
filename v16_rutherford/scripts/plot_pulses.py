import matplotlib.pyplot as plt
import pandas as pd


def read(filename):
    df = pd.read_csv(
        filename,
        skipinitialspace=True,
        header=None,
        usecols=[3, 4],
        names=['time', 'voltage'],
    )

    return df


def main():

    unamp = read(
        'data/pulse_without_amp/F0000CH1.CSV',
    )
    amp = read(
        'data/pulse_with_amp/F0001CH1.CSV',
    )

    plt.subplot(2, 1, 1)
    plt.plot(unamp.time * 1000, unamp.voltage)
    plt.xlabel('Zeit in Millisekunden')
    plt.ylabel('Spannung in Volt')

    plt.subplot(2, 1, 2)
    plt.plot(amp.time * 1000, amp.voltage)

    plt.xlabel('Zeit in Millisekunden')
    plt.ylabel('Spannung in Volt')

    plt.tight_layout(pad=0)
    plt.savefig('build/plots/pulses.pdf')



if __name__ == '__main__':
    main()
