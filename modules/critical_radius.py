"""
Module that determines the critical radius where eccentricity becomes larger,
using the data from Kennefick (1998)

author: Matthias Fabry
date: 26 May 2019
"""

import numpy as np

data = np.array([[-0.9, -0.5, 0.0, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                [9.64, 8.37, 6.68, 4.70, 3.76, 2.56, 2.03, 1.47, 1.0]])


def r_crit(a):
    """
    Linearly interpolates the data of Kennefick to get the critical radius for a given spin
    :param a: float, spin of the BH
    :return: float, the critical radius
    """
    if a < -0.9:
        raise ValueError('spin too low for data of Kennefick')
    i = 0
    while a >= data[0][i]:
        if a == data[0][i]:
            return data[1][i]
        i += 1
    fit = np.polyfit(data[0][i-1:i+1], data[1][i-1:i+1], 1)
    return np.polyval(fit, a)
