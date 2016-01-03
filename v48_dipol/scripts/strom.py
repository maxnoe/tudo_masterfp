import matplotlib.style
matplotlib.style.use('../matplotlibrc')
import matplotlib.pyplot as plt
import numpy as np

T = np.linspace(0.1, 5, 1000)
y = np.exp(- 1/T)

plt.semilogy(1/T, y, label=r'$j(T) \sim e^{-\frac{1}{T}}$', color="#3785ed")
plt.xlabel(r'$\frac{1}{T}$')
plt.ylabel(r'$ j(T) $$')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/strom.pdf')
