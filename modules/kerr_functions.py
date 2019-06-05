"""
Equatorial Kerr functions.
Allows for easy solving of the kerr geodesic equations.

author: Matthias Fabry
date: 6 May 2019
"""

import numpy as np
import numpy.polynomial.polynomial as poly


def delta(a, r):
    return r**2-2*r+a**2


def lambda_kerr(a, r):
    return (r**2+a**2)**2-a**2*delta(a, r)


def _t_kerr(a, r, en, l_z):
    return en*(r**2+a**2) - l_z*a


def v_r(a, r, en, l_z):
    return _t_kerr(a, r, en, l_z)**2 - delta(a, r)*(r**2 + (l_z - a*en)**2)


def v_r_coeffs(a, en, l_z):
    ome2 = 1.-en**2
    ae_m_lz = a * en - l_z
    return [0.,
            2. * ae_m_lz ** 2 / ome2,
            -(a * a * ome2 + l_z ** 2) / ome2,
            2. / ome2,
            -1.]


def v_t(a, r, en, l_z):
    return -a * (a * en - l_z) + (r ** 2 + a ** 2) * _t_kerr(a, r, en, l_z) / delta(a, r)


def v_phi(a, r, en, l_z):
    return -(a * en - l_z) + a * _t_kerr(a, r, en, l_z) / delta(a, r)


def drdtau(a, r, e, l_z):
    return 1 / r ** 2 * v_r(a, r, e, l_z) ** 0.5


def dtdtau(a, r, e, l_z):
    return 1/r**2*v_t(a, r, e, l_z)


def dphidtau(a, r, e, l_z):
    return 1/r**2*v_phi(a, r, e, l_z)


def radial_roots(a, en, l_z):
    p = poly.Polynomial(v_r_coeffs(a, en, l_z))
    roots = p.roots()
    return np.real(roots)


def get_orbital_parameters(a, en, l_z):
    """
    determines the orbital parameters (semi-latus rectum, eccentricity) given an energy and angular momentum
    :param a: float, spin of BH
    :param en: float, energy
    :param l_z: float, angular momentum
    :return: float, float; semi-latus rectum and eccentricity
    """
    roots = radial_roots(a, en, l_z)
    eccentricity = abs(roots[3]-roots[2])/(roots[3]+roots[2])
    semilatusrectum = 2*roots[3]*roots[2]/(roots[3]+roots[2])
    return semilatusrectum, eccentricity


get_orbital_parameters = np.vectorize(get_orbital_parameters)


def _x2(a, p, e):
    if a >= 0:
        sign = 1
    else:
        sign = -1
    return (-_n(a, p, e) - sign*np.sqrt(_deltax(a, p, e))) / (2*_f(a, p, e))


def _f(a, p, e):
    return 1.0/p**3*(p**3 - 2*(3+e**2)*p**2 + (3+e**2)**2*p - 4*a**2*(1-e**2)**2)


def _n(a, p, e):
    return 2.0/p*(-p**2+(3+e**2-a**2)*p-a**2*(1+3*e**2))


def _deltax(a, p, e):
    return 16*a**2/p**3*(p**4 - 4*p**3 + 2*(2*(1-e**2)+a**2*(1+e**2))*p**2 - 4*a**2*(1-e**2)*p + a**4*(1-e**2)**2)


def energy(a: float, p: float, e: float):
    """
    Energy of an equatorial orbit in Kerr
    :param a: float, spin of Kerr BH
    :param p: float, semi-latus rectum
    :param e: float, eccentricity
    :return: float, energy
    """
    return np.sqrt(1 - (1-e**2) / p * (1 - _x2(a, p, e) / p**2 * (1 - e**2)))


def lz(a: float, p: float, e: float):
    """
    Angular momentum for an eccentric orbit
    :param a: float, spin of Kerr BH
    :param p: float, semi-latus rectum
    :param e: float, eccentricity
    :return: float, angular momentum
    """
    if a >= 0:
        sign = 1
    else:
        sign = -1
    return a*energy(a, p, e)+sign*np.sqrt(_x2(a, p, e))


def lz_circ(a: float, r: float) -> float:
    """
    Angular momentum for a circular orbit
    :param a: float, spin of Kerr BH
    :param r: float, radius of orbit
    :return: float, angular momentum
    """
    return ((r ** 2 - 2 * a * r ** 0.5 + a ** 2) /
            (r ** 0.75 * (r ** 1.5 - 3 * r ** 0.5 + 2 * a) ** 0.5))
