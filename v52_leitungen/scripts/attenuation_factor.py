import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks_cwt
from IPython import embed


def peakdet(v, delta, x=None):
    """
    see https://gist.github.com/endolith/250860
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

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
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


if __name__ == '__main__':

    rectangle_short = pd.read_csv(
        './data/att_short_102_9kHz_fft.csv',
        header=2, names=['f', 'A']
    )
    rectangle_long = pd.read_csv(
        './data/att_long_102_9kHz_fft.csv',
        header=2, names=['f', 'A']
    )

    peaks_long, _ = peakdet(rectangle_long['A'].values, 10)
    peaks_short, _ = peakdet(rectangle_short['A'].values, 10)
    print(peaks_short.shape)
    print(peaks_long.shape)

    fig, ax = plt.subplots()
    ax.plot('f', 'A', data=rectangle_long, label='85m-Kabel')
    ax.plot('f', 'A', data=rectangle_short, label='kurzes Kabel')

    ax.plot('f', 'A', '.', data=rectangle_long.iloc[peaks_long[:, 0]], label='85m-Kabel')
    ax.plot('f', 'A', '.', data=rectangle_short.iloc[peaks_short[:, 0]], label='kurzes Kabel')

    ax.legend()
    plt.show()
