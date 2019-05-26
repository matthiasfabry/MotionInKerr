"""
Evaluation of energy, angular momentum and angular velocity of the ISCO

author: Matthias Fabry
date: 26 May 2019
"""
import numpy as np
from modules.orbits import risco


def e_isco(a):
    return (1 - 2 / risco(a) + a / risco(a) ** 1.5) / np.sqrt(1 - 3 / risco(a) + 2 * a / risco(a) ** 1.5)


def omega_isco(a):
    return 1 / (risco(a) ** 1.5 + a)


def lz_isco(a):
    return 2 * (3 * np.sqrt(risco(a)) - 2 * a) / np.sqrt(3 * risco(a))
