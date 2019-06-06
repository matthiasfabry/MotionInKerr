# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from modules.orbits import *

plt.rc('text', usetex=True)
font = {'family': 'serif', 'size': 16}
plt.rc('font', **font)


aes = np.linspace(0, 1, 500)
plt.figure(figsize=(7, 4), dpi=200)
plt.plot(aes, rplus(aes), label=r'outer horizon')
plt.plot(aes, rph(aes), label=r'photon orbit')
plt.plot(aes, rmb(aes), label=r'marginally bound')
plt.plot(aes, risco(aes), label=r'ISCO')
plt.plot(aes, np.ones(len(aes))*rergo(), label=r'ergoradius')
plt.legend()
plt.ylabel(r'$r/M$')
plt.xlabel(r'$a/M$')
plt.tight_layout(pad=0.2)
plt.show()
