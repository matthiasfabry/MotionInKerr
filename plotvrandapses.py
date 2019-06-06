# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from modules.kerr_functions import v_r, energy, lz, radial_roots
import modules.kerr_functions as kerr

plt.rc('font', **{'family': 'serif'})
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})


num = 10000
rs = np.linspace(1, 5, num)
a = 0.99
p = 2.2
e = 0.5
en = energy(a, p, e)
lz = lz(a, p, e)
roots = radial_roots(a, en, lz)
ys = v_r(a, rs, en, lz)/rs**4
plt.figure(figsize=(7, 4), dpi=200)
plt.plot(rs, ys, 'k')
plt.hlines(0, 1, 5, linestyles='dotted')
plt.ylim(-0.04, 0.08)
indices = np.where(np.logical_and(rs > roots[2], ys > 0))
plt.fill_between(rs[indices], np.zeros(num)[indices], ys[indices], color='black', alpha='0.4')
plt.xlabel(r'$r/M$')
plt.ylabel(r'$V_r/r^4$')
arrow = dict(facecolor='black', arrowstyle='->')
plt.annotate(r'$r_a$', xy=(roots[3], 0), xytext=(3.3, -0.03), arrowprops=arrow)
plt.annotate(r'$r_p$', xy=(roots[2], 0), xytext=(2.3, -0.03), arrowprops=arrow)
plt.tight_layout(pad=0.2)
plt.show()
