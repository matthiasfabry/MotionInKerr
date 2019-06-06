# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from modules.eob_inspiral import EOBInspiral

plt.rc('text', usetex=True)
font = {'family': 'serif', 'size': 16}
plt.rc('font', **font)


inspiral = EOBInspiral(0.8, 1e-6, 1.2, correlation_mode='iscopercent')

plt.figure(figsize=(7, 4), dpi=200)
inspiral.plot_crossing_correlation()
plt.tight_layout(pad=0.2)
# inspiral2 = EOBInspiral(0.9, 1e-2, 1.2, correlation_mode='scalewithX')
#
# plt.figure(figsize=(7, 4), dpi=200)
# inspiral2.plot_crossing_correlation()

plt.show()