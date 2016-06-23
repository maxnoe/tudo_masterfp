import matplotlib.pyplot as plt
import numpy as np

color = [e['color'] for e in plt.rcParams['axes.prop_cycle']]

if __name__ == '__main__':
    fig, ax = plt.subplots()
    width, height = fig.get_size_inches()
    fig.set_size_inches(height, height)

    ax.set_xticks([0, 10, 20])
    ax.set_xlabel(r'$l \mathbin{/} \si{\meter}$')
    ax.set_ylabel(r'$t \mathbin{/} T$')

    ax.set_yticks(np.arange(7))

    d = 0.03

    linestyle = (0, (4, 1))

    ax.plot([10, 20 -d, 20-d, 0], [3+d, 4+d, 4-d, 6-d], color=color[0], linestyle=linestyle)
    ax.plot([10, 0], [5 + 2*d, 6 + 2*d], color=color[1], linestyle=linestyle)

    ax.plot([0, 20, 0, 20], [0, 2, 4, 6], color=color[0])
    ax.plot([10, 0, 20, 0], [1, 2, 4, 6], color=color[1])
    ax.plot([10, 0, 20], np.array([3, 4, 6]) + d, color=color[1])
    ax.plot([10, 0], [5 + d, 6 + d], color=color[0], linestyle=linestyle)
    ax.plot([10, 20], [5 + 2*d, 6 + 2*d], color=color[0], linestyle=linestyle)
    ax.plot([10, 20], [5 - d, 6 - d], color=color[1], linestyle=linestyle)


    ax.axvline(10, color='black')
    ax.axvline(20, color='black')

    fig.tight_layout(pad=0)
    fig.savefig('build/impuls.pdf')
