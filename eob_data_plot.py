import matplotlib.pyplot as plt
import numpy as np

spin = -0.7
eta = 1e-4
risco_factor = 1.01
data = np.loadtxt('data_saves/EOBInspiral_{}_{}_{}.txt'.format(spin, eta, risco_factor), skiprows=1, unpack=True)

plt.figure()
plt.plot(data[0], data[1])
plt.title('r(t)')
plt.figure()
plt.plot(data[0], data[2])
plt.title('phi(t)')
plt.figure()
plt.plot(data[0], data[3])
plt.title('pr(t)')
plt.figure()
plt.plot(data[0], data[4])
plt.title('pphi(t)')
plt.figure()
plt.plot(data[0], data[5])
plt.title('nkF(t)')
plt.figure()
plt.plot(data[0], data[6])
plt.title('nkRadialForce(t)')
plt.figure()
plt.plot(data[0], data[7])
plt.title('kRadialForce(t)')
plt.figure()
plt.plot(data[0], data[8])
plt.title('dphidt(t)')
plt.show()