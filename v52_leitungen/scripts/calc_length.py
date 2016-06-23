from ruamel import yaml
import scipy.constants as const
from pint import UnitRegistry
import numpy as np

u = UnitRegistry()

c0 = const.c * u.meter / u.second

epsilon_r = 2.25


def calc_length(delta_t):
    return c0 * delta_t * 0.5 / np.sqrt(epsilon_r)


with open('data/length.yaml') as f:
    data = yaml.safe_load(f)


table_head = r'''\begin{tabular}{
  l
  S[table-format=3.1]
  S[table-format=2.2]
  }
  \toprule
  Kabel &
  {$\increment t \mathbin{/} \si{\nano\second}$} &
  {$l \mathbin{/} \si{\meter}$} \\
  \midrule
'''
table_foot = r'''  \bottomrule
\end{tabular}
'''

results = {}
with open('build/length.tex', 'w') as f:
    f.write(table_head)
    for key, val in sorted(data.items(), key=lambda x: x[0]):

        delta_t = (val['t2'] - val['t1']) * u.nanosecond

        length = calc_length(delta_t)
        results[key] = {'length': float(length.to(u.meter).magnitude)}

        print(key)
        print(3 * ' ', delta_t)
        print(3 * ' ', length.to(u.meter))

        f.write('  ' + key + ' & ')
        f.write('  {:3.1f} & '.format(delta_t.to('ns').magnitude))
        f.write('  {:2.2f} \\\\\n'.format(length.to('m').magnitude))

    f.write(table_foot)

with open('build/length.yaml', 'w') as f:
    f.write(yaml.dump(results))
