import modules.eob_inspiral as insp
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
font = {'family': 'serif', 'size': 16}
plt.rc('font', **font)

orbit = insp.EOBInspiral(0.9, 1e-6, 1.15, stop_factor=0.99)
plt.figure(figsize=(7, 4), dpi=200)
plt.ticklabel_format(axis='y', style='sci')
orbit.plot_eccentricity_r()
plt.tight_layout(pad=0.2)
plt.show()
