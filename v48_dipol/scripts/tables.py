import pandas as pd
import scipy.constants as const


data = pd.read_csv(
    'data/messwerte_2.csv',
    skiprows=(1, 2, 3, 4, 5),
)
data['T'] = data['T'].apply(const.C2K)


table_top = r'''\begin{tabular}{
        S[table-format=2.1]
        S[table-format=4.0]
        S[table-format=3.1]
    }
    \toprule
    {$t \mathbin{/} \si{\min}$} &
    {$I \mathbin{/} \si{\pico\ampere}$} &
    {$T \mathbin{/} \si{\kelvin}$} \\
    \midrule
'''

table_bottom = r'''    \bottomrule
\end{tabular}
'''

n_rows = len(data.index)

for i in range(3):
    with open('build/data_{}.tex'.format(i + 1), 'w') as f:

        f.write(table_top)

        low = i * n_rows // 3
        up = (i + 1) * n_rows // 3

        for row in data.iloc[low:up].itertuples():

            f.write(
                '{row.t:2.1f} & {row.I:4.0f} & {row.T:3.1f}'.format(
                    row=row
                ) + r'\\' + '\n'
            )

        f.write(table_bottom)
