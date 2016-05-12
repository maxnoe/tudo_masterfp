import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def peakdet(v, delta, x=None):
    """
    see https://gist.github.com/endolith/250860
    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        raise ValueError('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        raise ValueError('Input argument delta must be a scalar')

    if delta <= 0:
        raise ValueError('delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


if __name__ == '__main__':

    fft_short = pd.read_csv(
        './data/att_short_102_9kHz_fft.csv',
        header=2, names=['f', 'A']
    )
    fft_long = pd.read_csv(
        './data/att_long_102_9kHz_fft.csv',
        header=2, names=['f', 'A']
    )

    fft_long['f'] /= 1e6
    fft_short['f'] /= 1e6

    peaks_long, _ = peakdet(fft_long['A'].values, 11)
    peaks_short, _ = peakdet(fft_short['A'].values, 11)

    assert len(peaks_long) == len(peaks_short)

    attenuation = fft_long['A'].loc[peaks_long].values - fft_short['A'].loc[peaks_short].values

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot('f', 'A', data=fft_long, label='85m-Kabel')
    ax1.plot('f', 'A', data=fft_short, label='kurzes Kabel')

    ax1.plot(
        'f', 'A', '.',
        data=fft_long.iloc[peaks_long], label='85m-Kabel'
    )
    ax1.plot(
        'f', 'A', '.',
        data=fft_short.iloc[peaks_short],
        label='kurzes Kabel',
    )

    ax1.legend(ncol=2)
    ax1.set_ylabel(r'$A \mathbin{/} \si{\deci\bel}$')

    ax1.set_xlim(0, 5)
    ax1.set_ylim(-35, 25)

    print(fft_long['f'].loc[peaks_long].values.shape)
    print(attenuation.shape)

    ax2.plot(fft_long['f'].loc[peaks_long], attenuation, '.')
    ax2.set_xlabel(r'$f \mathbin{/} \si{\mega\hertz}$')
    ax2.set_ylabel(r'$α \mathbin{/} \si{\deci\bel}$')

    ax2.set_ylim(-6, 0.5)


    fig.tight_layout(pad=0.3)
    fig.savefig('build/attenuation.pdf')
