import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

plt.rc('text', usetex=True)
font = {'family': 'serif', 'size': 14}
plt.rc('font', **font)


data = np.loadtxt('parameterspacesearch_scale.txt', skiprows=1)
etas = data[0, 1:]
spins = data[1:, 0]
etas_grid, spins_grid = np.meshgrid(np.log10(etas), spins)

maximum = max(max(data[i, 1:]) for i in range(1, len(data)))


def threedee_fig():
    fig = plt.figure(figsize=(7, 4), dpi=200)
    ax = fig.gca(projection='3d')
    ax.plot_surface(etas_grid, spins_grid, np.log10(data[1:, 1:]), cmap='inferno')
    # ax.set_zlim(0, 0.8*maximum)
    ax.set_xlabel(r'$\log\eta$')
    ax.set_ylabel(r'$a/M$')
    ax.set_zlabel('log(average relative error)')
    ax.set_xticks([-2, -3, -4, -5, -6])
    ax.set_yticks([-0.8, -0.4, 0.0, 0.4, 0.8])
    ax.set_zticks([-1*i for i in range(2, 5)])
    plt.tight_layout()
    plt.show()


def spin_slice(i):
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(spins, data[1:, i])
    ax.set_xlabel(r'$a/M$')
    ax.set_ylabel(r'average relative error')
    plt.show()


def eta_slice(i):
    plt.figure()
    ax = plt.subplot(111)
    ax.semilogx(etas, data[i, 1:])
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'average relative error')
    ax.annotate(r'$a/M = {}$'.format(np.round(spins[i-1], 3)), (0.05, 0.85), xycoords='axes fraction')
    plt.tight_layout(pad=0.2)
    plt.show()


threedee_fig()
