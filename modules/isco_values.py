"""
Created on Apr 12, 2019

@author: Matthias
"""
import numpy as np


def risco(a):
    if a > 0.0:
        x = 1.
    else:
        x = -1
    z1 = 1 + np.cbrt(1 - a ** 2) * (np.cbrt(1 + a) + np.cbrt(1 - a))
    z2 = np.sqrt(3 * a ** 2 + z1 ** 2)
    return 3 + z2 - x * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def e_isco(a):
    return (1 - 2 / risco(a) + a / risco(a) ** 1.5) / np.sqrt(1 - 3 / risco(a) + 2 * a / risco(a) ** 1.5)


def omega_isco(a):
    return 1 / (risco(a) ** 1.5 + a)


def lz_isco(a):
    return 2 * (3 * np.sqrt(risco(a)) - 2 * a) / np.sqrt(3 * risco(a))
