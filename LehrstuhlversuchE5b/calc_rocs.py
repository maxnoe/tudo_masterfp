import pandas as pd
import numpy as np
import os


infiles = [
    './build/AdaBoost.hdf5',
    './build/RandomForest.hdf5',
    './build/NaiveBayes.hdf5',
]

header = r'''\begin{tabular}{l S[table-format=1.4] @{${}\pm{}$} S[table-format=1.4]}
\toprule
Algorithmus & \multicolumn{2}{c}{$A_\mathrm{ROC}$} \\
\midrule
'''
footer = r'''\bottomrule
\end{tabular}
'''

with open('build/roc_table.tex', 'w') as f:
    f.write(header)
    for infile in infiles:
        perf = pd.read_hdf(infile, 'performance')
        name = os.path.splitext(os.path.basename(infile))[0]

        roc_aucs = []
        for n_cv, df in perf.iteritems():
            df = df.sort_values(by='fpr')
            roc_aucs.append(np.trapz(df['tpr'], df['fpr']))

        f.write(name)
        f.write(' & ')
        f.write('{:1.4f}'.format(np.mean(roc_aucs)))
        f.write(' & ')
        f.write('{:1.4f}'.format(np.std(roc_aucs)))
        f.write(r' \\')
        f.write('\n')

    f.write(footer)

with open('build/roc_table_feature_selection.tex', 'w') as f:
    f.write(header)
    for infile in infiles:
        perf = pd.read_hdf(infile, 'performance_feature_selection')
        name = os.path.splitext(os.path.basename(infile))[0]

        roc_aucs = []
        for n_cv, df in perf.iteritems():
            df = df.sort_values(by='fpr')
            roc_aucs.append(np.trapz(df['tpr'], df['fpr']))

        f.write(name)
        f.write(' & ')
        f.write('{:1.4f}'.format(np.mean(roc_aucs)))
        f.write(' & ')
        f.write('{:1.4f}'.format(np.std(roc_aucs)))
        f.write(r' \\')
        f.write('\n')

    f.write(footer)
