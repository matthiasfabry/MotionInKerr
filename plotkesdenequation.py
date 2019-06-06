import matplotlib.pyplot as plt

from modules.kesden_expansion import KesdensExpansionDimensionless

plt.rc('text', usetex=True)
font = {'family': 'serif', 'size': 16}
plt.rc('font', **font)

kesdendimless = KesdensExpansionDimensionless()

kesdendimless.plot_data()

plt.show()
