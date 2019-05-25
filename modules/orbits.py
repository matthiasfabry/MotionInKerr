# important orbits
import numpy as np


def rplus(a):
    return 1 + np.sqrt(1-a**2)


def rph(a):
    return 2*(1+np.cos((2./3)*np.arccos(-a)))


def rmb(a):
    return 2-a+2*np.sqrt(1-a)


def risco(a):
    if a > 0:
        x = 1.
    else:
        x = -1
    z1 = 1+np.cbrt(1-a**2)*(np.cbrt(1+a)+np.cbrt(1-a))
    z2 = np.sqrt(3*a**2+z1**2)
    return 3+z2-x*np.sqrt((3-z1)*(3+z1+2*z2))


risco = np.vectorize(risco)


def rergo():
    return 2.0
