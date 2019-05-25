"""
Equatorial Kerr potential functions.
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


def t_kerr(a, r, en, lz):
    return en*(r**2+a**2)-lz*a


def v_r(a, r, en, lz):
    return t_kerr(a, r, en, lz)**2-delta(a, r)*(r**2+(lz-a*en)**2)


def v_r_coeffs(a, en, lz):
    ome2 = 1.-en**2
    ae_m_lz = a*en - lz
    return [0.,
            2.*ae_m_lz**2/ome2,
            -(a*a*ome2 + lz**2)/ome2,
            2./ome2,
            -1.]


def v_t(a, r, e, lz):
    return -a*(a*e-lz)+(r**2+a**2)*t_kerr(a, r, e, lz)/delta(a, r)


def v_phi(a, r, e, lz):
    return -(a*e-lz)+a*t_kerr(a, r, e, lz)/delta(a, r)


def drdtau(a, r, e, lz):
    return 1/r**2*v_r(a, r, e, lz)**0.5


def dtdtau(a, r, e, lz):
    return 1/r**2*v_t(a, r, e, lz)


def dphidtau(a, r, e, lz):
    return 1/r**2*v_phi(a, r, e, lz)


def lz_circ(a, r):
    return ((r ** 2 - 2 * a * r ** 0.5 + a ** 2) /
            (r ** 0.75 * (r ** 1.5 - 3 * r ** 0.5 + 2 * a) ** 0.5))


def radial_roots(a, en, lz):
    p = poly.Polynomial(v_r_coeffs(a, en ,lz))
    roots = p.roots()
    return np.real(roots)


def get_orbital_params(a, en, lz):
    roots = radial_roots(a, en, lz)
    eccentricity = abs(roots[3]-roots[2])/(roots[3]+roots[2])
    semilatusrectum = 2*roots[3]*roots[2]/(roots[3]+roots[2])
    return semilatusrectum,eccentricity


getOrbitalParams = np.vectorize(get_orbital_params)


# energy for equatorial orbits
def energy(p, e, a):
    return np.sqrt(1-(1-e**2)/p*(1-_x2(p, e, a)/p**2*(1-e**2)))


def _x2(p, e, a):
    return (-_n(p, e, a)-np.sqrt(_deltax(p, e, a)))/(2*_f(p, e, a))


def _f(p, e, a):
    return 1.0/p**3*(p**3-2*(3+e**2)*p**2+(3+e**2)**2*p-4*a**2*(1-e**2)**2)


def _c(p, e, a):
    return (a**2-p)**2


def _n(p, e, a):
    return 2.0/p*(-p**2+(3+e**2-a**2)*p-a**2*(1+3*e**2))


def _deltax(p, e, a):
    return _n(p, e, a)**2-4*_f(p, e, a)*_c(p, e, a)


# ang mom for equatorial orbits
def lz(p, e, a):
    return a*energy(p, e, a)+np.sqrt(_x2(p, e, a))
