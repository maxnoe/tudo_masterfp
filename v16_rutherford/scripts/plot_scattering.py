import matplotlib.pyplot as plt
import pandas as pd

df  = pd.read_csv('data/scattering_gold_2_micrometer.csv', comment='#', index_col=0)
plt.plot(df.index, df.counts, '+')
plt.xlabel('Winkel in Grad')
plt.ylabel('Streuquerschnitt')

# df  = pd.read_csv('data/scattering_gold_4_micrometer.csv', comment='#', index_col=0)
# plt.plot(df.index, df.counts, '+', label='4 ')
plt.tight_layout()
plt.savefig('build/plots/scattering.pdf')
