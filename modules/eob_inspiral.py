"""
Class supporting the calculation of Effective One Body inspirals, following the recipe of Damour et al.

author: Matthias Fabry
date: 25 May 2019

"""

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import sympy as sp
import modules.kesden_expansion

from matplotlib import animation
from modules.corrections import eps_at_isco
from modules.isco_values import *
from modules.kerr_functions import *
from modules.orbits import rplus, rergo
from modules.critical_radius import r_crit


class EOBInspiral:

    # Helper functions derived from the kerr metric
    def _betaphi(self, r):
        return self._gtphi(r) / self._gtt(r)

    def _alpha(self, r):
        return 1 / ((-self._gtt(r)) ** 0.5)

    def _gammarr(self, r):
        return self._grr(r)

    def _gammaphiphi(self, r):
        return self._gphiphi(r) - (self._gtphi(r) ** 2) / self._gtt(r)

    # Kerr metric components
    def _gtt(self, r):
        return -lambda_kerr(self._a, r) / (r ** 2 * delta(self._a, r))

    def _grr(self, r):
        return delta(self._a, r) / r ** 2

    def _gphiphi(self, r):
        return 1/lambda_kerr(self._a, r)*(r**2-4*self._a**2/delta(self._a, r))

    def _gtphi(self, r):
        return -2 * self._a / (r * delta(self._a, r))

    # EOB derivatives
    def _drdt(self, r, pr, pphi):
        return delta(self._a, r) / (r ** 2 + self._a ** 2) * self._dHdprlamb(r, pr, pphi)

    def _dphidt(self, r, pr, pphi):
        return self._dHdpphilamb(r, pr, pphi)

    def _dprdt(self, r, pr, pphi):
        if self._doLoss:
            return -delta(self._a, r) / (r ** 2 + self._a ** 2) * self._dHdrlamb(r, pr, pphi) \
                   - self._f(r, pr, pphi) * pr / pphi
        else:
            return -delta(self._a, r) / (r ** 2 + self._a ** 2) * self._dHdrlamb(r, pr, pphi)

    def _dpphidt(self, r, pr, pphi):
        if self._doLoss:
            return self._f(r, pr, pphi)
        else:
            return 0.0

    def _f(self, r, pr, pphi):
        """
        This is the expression for the energy flux dE/dt. Currently the ISCO flux of Finn & Thorne is used
        """
        return -1. / self._dphidt(r, pr, pphi) * 32 / 5 * self._eta * omega_isco(self._a) ** (10./3) * eps_at_isco(self._a)

    def _ders_proper(self, t, y):
        dr = self._drdt(y[0], y[2], y[3]) * dtdtau(self._a, y[0], self._Hlamb(y[0], y[2], y[3]), y[3])
        dphi = self._dphidt(y[0], y[2], y[3]) * dtdtau(self._a, y[0], self._Hlamb(y[0], y[2], y[3]), y[3])
        dpr = self._dprdt(y[0], y[2], y[3]) * dtdtau(self._a, y[0], self._Hlamb(y[0], y[2], y[3]), y[3])
        dpphi = self._dpphidt(y[0], y[2], y[3]) * dtdtau(self._a, y[0], self._Hlamb(y[0], y[2], y[3]), y[3])
        return [dr, dphi, dpr, dpphi]

    def _ders_coordinate(self, t, y):
        dr = self._drdt(y[0], y[2], y[3])
        dphi = self._dphidt(y[0], y[2], y[3])
        dpr = self._dprdt(y[0], y[2], y[3])
        dpphi = self._dpphidt(y[0], y[2], y[3])
        return [dr, dphi, dpr, dpphi]

    def _evolve(self):
        """
        Evaluates the evolution of the inspiral.
        Depending on the various switches, event tracking during the integration is different.
        The integration will reset until at least r=r_isco is reached.
        """
        def transitions(t, y):
            return y[0] - self._r_isco

        if self._do_proper:
            ders = self._ders_proper
        else:
            ders = self._ders_coordinate

        if self._do_correlation:
            def transition_start(t, y):
                return y[0] - 1.01 * self._r_isco

            def transition_end(t, y):
                return y[0] - 0.99 * self._r_isco
            transition_end.terminal = True
            start = 0
            events = [transitions, transition_start, transition_end]
        else:
            if self._stop_factor is not None:
                def force_stop(t, y):
                    return y[0] - self._stop_factor * self._r_isco

                force_stop.terminal = True
                start = 0
                events = [transitions, force_stop]
            else:
                def plunges(t, y):
                    return y[0] - 1.01 * rplus(self._a)
                plunges.terminal = True
                start = 0
                events = [transitions, plunges]

        if self._ecc_init != 0:
            lz_init = lz(self._a, self._r_init, self._ecc_init)
        else:
            lz_init = lz_circ(self._a, self._r_init)
        done = False
        count = 1
        integration = None
        while not done:
            print('starting an eob integration, number', count, 'of the same inspiral')
            # noinspection PyTypeChecker
            integration = integrate.solve_ivp(ders,
                                              (start, self._span),
                                              (self._r_init, 0, 0, lz_init),
                                              method='Radau', dense_output=True,
                                              events=events)
            if self._do_correlation:
                if len(integration.t_events[2]) > 0:
                    done = True
                else:
                    self._span *= 2.0
                    count += 1
            else:
                if len(integration.t_events[0]) > 0:
                    done = True
                else:
                    self._span *= 2.0
                    count += 1
        # print(integration)
        self._transition_time = integration.t_events[0][0]
        if self._do_correlation:
            ts = None
            if self._correlation_mode == 'eob':
                ts = np.linspace(integration.t_events[1][0], integration.t_events[2][0], 200, endpoint=True) - \
                     (integration.t_events[0][0] - self._kesden_equivalent.get_isco_crossing_time())
                evals = integration.sol.__call__(ts)
                return ts, evals[0], evals[1], evals[2], evals[3]
            elif self._correlation_mode == 'mk':
                ts = self._kesden_equivalent.get_taus() + \
                     (integration.t_events[0][0] - self._kesden_equivalent.get_isco_crossing_time())
                evals = integration.sol.__call__(ts)
                ts -= (integration.t_events[0][0] - self._kesden_equivalent.get_isco_crossing_time())
                return ts, evals[0], evals[1], evals[2], evals[3]
            if ts is None:
                raise RuntimeError('correlation mode is not set properly')
        else:
            ts = np.linspace(0, integration.t[-1], int(1e4))
            evals = integration.sol.__call__(ts)
            return ts, evals[0], evals[1], evals[2], evals[3]

    def __init__(self, a: float, eta: float, risco_factor: float, eccentricity: float = 0,
                 stop_factor: float = None, do_loss: bool = True,
                 do_proper: bool = False,
                 correlation_mode: str = None) -> None:
        """
        Creates an EOB inspiral object with the paramters given. It immediately evolves the inspiral.
        Also, if do_correlation is True, it creates a KesdenExpansion object to compare with
        :param a: float, spin of Kerr BH -1<a<1
        :param eta: float, mass ratio
        :param risco_factor: float, start of the inspiral in units of r_isco
        :param eccentricity: float, starting eccentricity
        :param stop_factor: float, stop of the inspiral in units of r_isco
        :param do_loss: boolean, switch to put energy flux to zero if False
        :param do_proper: boolean, switch to do inspiral in terms of proper time (True) or coordinate time (False)
        :param correlation_mode: string, switch to do a correlation with a KesdenInspiral object
        """
        self._ecc_init = eccentricity
        self._stop_factor = stop_factor
        if correlation_mode is not None:
            self._do_correlation = True
            self._correlation_mode = correlation_mode
            self._kesden_equivalent = modules.kesden_expansion.KesdensExpansion(self, a, eta)
        else:
            self._do_correlation = False
        self._do_proper = do_proper
        self._doLoss = do_loss
        self._span = 10000.0
        self._risco_factor = risco_factor
        self._a = a
        self._eta = eta
        self._r_isco = risco(self._a)
        self._r_init = risco_factor * self._r_isco
        r, pr, pphi = sp.symbols('r pr pphi')
        self._H = self._betaphi(r) * pphi + self._alpha(r) * \
            (1 + self._gammarr(r) * pr**2 + self._gammaphiphi(r) * pphi**2) ** 0.5
        self._dHdpphi = self._H.diff(pphi)
        self._dHdpr = self._H.diff(pr)
        self._dHdr = self._H.diff(r)
        self._Hlamb = sp.lambdify((r, pr, pphi), self._H, 'numpy')
        self._dHdprlamb = sp.lambdify((r, pr, pphi), self._dHdpr, 'numpy')
        self._dHdpphilamb = sp.lambdify((r, pr, pphi), self._dHdpphi, 'numpy')
        self._dHdrlamb = sp.lambdify((r, pr, pphi), self._dHdr, 'numpy')

        if correlation_mode == 'oeb':
            self._ts, self._rs, self._phis, self._prs, self._pphis = self._evolve()
            self._kesden_equivalent.evaluate_in(self._ts)
        elif correlation_mode == 'mk':
            self._kesden_equivalent.evaluate_in_transition()
            self._ts, self._rs, self._phis, self._prs, self._pphis = self._evolve()
        else:
            self._ts, self._rs, self._phis, self._prs, self._pphis = self._evolve()
        self._ens = self._Hlamb(self._rs, self._prs, self._pphis)
        self._ps, self._es = get_orbital_parameters(self._a, self._ens, self._pphis)
        self._plot_vr_calls = 0

    def get_ts(self):
        return self._ts

    def get_rs(self):
        return self._rs

    def get_es(self):
        return self._es

    def get_kesden_equivalent(self):
        return self._kesden_equivalent

    def get_isco_crossing_time(self):
        return self._transition_time

    def save_data(self):
        np.savetxt('data_saves/EOBInspiral_{}_{}_{}.txt'.format(self._a, self._eta, self._risco_factor),
                   np.c_[self._ts, self._rs, self._phis, self._prs, self._pphis,
                         self._f(self._rs, self._prs, self._pphis),
                         self._f(self._rs, self._prs, self._pphis)*self._prs/self._pphis,
                         delta(self._a, self._rs)/(self._rs**2+self._a**2) *
                         self._dHdrlamb(self._rs, self._prs, self._pphis),
                         self._dphidt(self._rs, self._prs, self._pphis),
                         self._ens,
                         ],
                   header='t, r, phi, pr, pphi, nkF, nkRadialRorce, keplerianRadialForce, dphidt,\
                            energy')

    def correlate_isco_crossings(self):
        if not self._do_correlation:
            raise ValueError('correlation not supported in \'do_correlation = False\' mode')
        relerror = 0
        for i in range(len(self._ts)):
            relerror += np.abs(self._rs[i] - self._kesden_equivalent.get_rs()[i]) / \
                     self._rs[i]
        return relerror / len(self._ts)

    def r_inc(self):
        es = self._es
        rs = self._rs
        eprimes = [0] + [(es[i]-es[i-1])/(rs[i]-rs[i-1]) for i in range(1, len(rs))]
        radii = []
        for i in range(len(self._rs)):
            if -eprimes[i]/es[i] > 1:
                radii.append(self._rs[i])
        return radii

    def plot_vr_at(self, r):
        self._plot_vr_calls += 1
        if r > self._rs[0]:
            print('radius is larger than starting point of the inspiral; choose between {} and {}; skipping'
                  .format(self._rs[0], self._rs[-1]))
            return
        elif r < self._rs[-1]:
            print('radius is smaller than end point of the inspiral; choose between {} and {}; skipping'
                  .format(self._rs[0], self._rs[-1]))
            return
        index = 0
        for i in range(len(self._rs)):
            if self._rs[i] < r:
                index = i
                break
        if self._rs[index] - r > r - self._rs[index - 1]:
            index -= 1
        vrs = v_r(self._a, self._rs, self._ens[index], self._pphis[index])
        plt.plot(self._rs, vrs)
        plt.hlines(0, self._rs[-1], self._rs[0])
        plt.vlines(self._r_isco, min(vrs), max(vrs), 'g')
        plt.scatter(self._rs[index], vrs[index])
        plt.xlabel(r'$r/M$')
        plt.ylabel(r'$V_r$')
        if self._plot_vr_calls == 1:
            bbox = dict(boxstyle='round', fc='w', ec='w', lw=0., alpha=0.9, pad=0.2)
            plt.annotate(r'$r_{ISCO}$', (self._r_isco, -300*vrs[index]), color='g', bbox=bbox)

    def plot_crossing_correlation(self):
        if not self._do_correlation:
            raise ValueError('not supported in \'do_correlation = False\' mode')
        plt.plot(self._ts-self._kesden_equivalent.get_isco_crossing_time(), self._rs, 'b', label='EOB Inspiral')
        plt.plot(self._kesden_equivalent.get_taus()-self._kesden_equivalent.get_isco_crossing_time(),
                 self._kesden_equivalent.get_rs(), 'r', label='MK\'s expansion')
        plt.plot(self._ts-self._kesden_equivalent.get_isco_crossing_time(), np.ones(len(self._ts))*self._r_isco, 'g')
        plt.fill_between(self._ts-self._kesden_equivalent.get_isco_crossing_time(),
                         self._kesden_equivalent.get_rs(), self._rs, color='black', alpha='0.2')
        bbox = dict(boxstyle='round', fc='w', ec='w', lw=0., alpha=0.9, pad=0.2)
        plt.annotate(r'$r_{ISCO}$', (self._ts[int(len(self._ts) * 0.85)] -
                                     self._kesden_equivalent.get_isco_crossing_time(), self._r_isco),
                     color='g', bbox=bbox)
        plt.annotate(r'$a = {} M$'.format(self._a), (0.05, 0.05), xycoords='axes fraction', bbox=bbox)
        plt.annotate(r'$\eta = {}$'.format(self._eta), (0.05, 0.15), xycoords='axes fraction', bbox=bbox)
        plt.legend(loc='center left')
        plt.xlabel(r'$\tau/M-\tau_{crossing}/M$')
        plt.ylabel(r'$r/M$')

    def plot_energy_r(self):
        plt.plot(self._rs, self._ens, 'b.')
        plt.xlabel(r'$r$')
        plt.ylabel(r'energy')

    def plot_orbit_cartesian(self):
        plt.figure()
        ax = plt.gcf().add_subplot(111)
        ax.plot(self._phis, self._rs, linewidth=0.75)
        ax.plot(self._phis, self._r_isco * np.ones(len(self._phis)), 'g')
        ax.plot(self._phis, rplus(self._a) * np.ones(len(self._phis)), 'k')
        ax.plot(self._phis, rergo() * np.ones(len(self._phis)), 'r')
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$r$')
        plt.tight_layout(pad=0.2)

    def plot_orbit(self):
        plt.figure()
        twopi = np.linspace(0, 2 * np.pi, 100, endpoint=True)
        ax = plt.gcf().add_subplot(111, projection='polar')
        ax.plot(self._phis, self._rs, linewidth=0.75)
        ax.plot(twopi, self._r_isco * np.ones(len(twopi)), 'g')
        ax.plot(twopi, rplus(self._a) * np.ones(len(twopi)), 'k')
        ax.plot(twopi, rergo() * np.ones(len(twopi)), 'r')
        bbox = dict(boxstyle='round', fc='w', ec='w', lw=0., alpha=0.9, pad=0.2)
        ax.annotate(r'$r_{ISCO}$', (4*np.pi / 10, self._r_isco), color='g', bbox=bbox)
        ax.annotate(r'$r_{ergo}$', (np.pi / 3, rergo()), color='r', bbox=bbox)
        ax.annotate(r'$r_{+}$', (-np.pi / 3, 0.7*rplus(self._a)), color='k', bbox=bbox)
        ax.annotate(r'$a = {} M$'.format(self._a), (0.05, 0.9), xycoords='figure fraction', bbox=bbox)
        ax.annotate(r'$\eta = {}$'.format(self._eta), (0.05, 0.8), xycoords='figure fraction', bbox=bbox)
        ax.annotate(r'$r_{{init}} = {} r_{{ISCO}}$'.format(self._risco_factor),
                    (0.05, 0.7), xycoords='figure fraction', bbox=bbox)
        ax.axis('off')
        plt.fill_between(twopi, 0, rplus(self._a), color='black')

    def plot_radial_trajectory(self):
        plt.plot(self._ts, self._rs, label='eob_inspiral')
        plt.plot(self._ts, np.ones(len(self._ts))*self._r_isco, 'g')
        if self._stop_factor is not None:
            plt.plot(self._ts, np.ones(len(self._ts))*self._r_isco*self._stop_factor)
        if self._do_proper:
            label = r'$\tau$'
        else:
            label = r'$t$'
        bbox = dict(boxstyle='round', fc='w', ec='w', lw=0., alpha=0.9, pad=0.2)
        plt.annotate(r'$r_{ISCO}$', (self._ts[int(len(self._ts)*0.85)], self._r_isco), color='g', bbox=bbox)
        plt.xlabel(label)
        plt.ylabel(r'$r/M$')

    def plot_eccentricity_t(self):
        plt.plot(self._ts, self._es)
        if self._do_proper:
            label = r'$\tau$'
        else:
            label = r'$t$'
        plt.xlabel(label)
        plt.ylabel(r'eccentricity')

    def plot_eccentricity_r(self):
        plt.plot(self._rs, self._es)
        plt.xlabel(r'$r/M$')
        plt.ylabel(r'$e$')
        plt.vlines(self._r_isco, 0, max(self._es), colors='g')
        bbox = dict(boxstyle='round', fc='w', ec='w', lw=0., alpha=0.9, pad=0.2)
        plt.annotate(r'$r_{ISCO}$', (self._r_isco, 0.1*max(self._es)), color='g', bbox=bbox)
        # plt.vlines(r_crit(self._a), min(self._es), max(self._es))
        # plt.annotate(r'$r_{crit}$', (r_crit(self._a), 0.5*max(self._es)), color='k', bbox=bbox)

    def plot_eccentricity_r_per_r_isco(self):
        plt.plot(self._rs/self._r_isco, self._es, label='a={}'.format(round(self._a,2)))
        plt.xlabel(r'$r/r_{isco}$')
        plt.ylabel(r'$e$')
        plt.vlines(1, 0, max(self._es), colors='g')
        plt.vlines(r_crit(self._a)/self._r_isco, min(self._es), max(self._es))
        bbox = dict(boxstyle='round', fc='w', ec='w', lw=0., alpha=0.9, pad=0.2)
        plt.annotate(r'$r_{crit}$', (r_crit(self._a)/self._r_isco, 0.5*max(self._es)), color='k', bbox=bbox)
        # rincs = self.r_inc()
        # plt.vlines([i/self._r_isco for i in rincs], min(self._es), max(self._es))


    def plot_eccentricity_p(self):
        plt.plot(self._ps, self._es)
        plt.xlabel(r'$p/M$')
        plt.ylabel(r'$e$')
        plt.vlines(self._r_isco, 0, max(self._es), colors='g')
        bbox = dict(boxstyle='round', fc='w', ec='w', lw=0., alpha=0.9, pad=0.2)
        plt.annotate(r'$r_{ISCO}$', (self._r_isco, 0.1*max(self._es)), color='g', bbox=bbox)
        plt.vlines(r_crit(self._a), min(self._es), max(self._es))
        plt.annotate(r'$r_{crit}$', (r_crit(self._a), 0.5*max(self._es)), color='k', bbox=bbox)

    def animate_orbit(self):
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111, projection='polar')
        ax.axis('off')
        xdata, ydata = [], []
        xpoint, ypoint = [], []
        ln, = plt.plot([], [], 'b-')
        pt, = plt.plot([], [], 'bo')
        plt.tight_layout(pad=0.2)
        two_pi = np.linspace(0, 2 * np.pi, 100, endpoint=True)

        def init():
            ax.set_ylim(0, self._rs[0])
            ax.plot(two_pi, self._r_isco * np.ones(len(two_pi)), 'g')
            ax.plot(two_pi, rplus(self._a) * np.ones(len(two_pi)), 'k')
            ax.plot(two_pi, rergo() * np.ones(len(two_pi)), 'r')
            ax.annotate(r'$r_{ISCO}$', (np.pi / 3, self._r_isco), color='g', fontsize=14)
            ax.annotate(r'$r_{ergo}$', (np.pi / 3, rergo()), color='r', fontsize=14)
            ax.annotate(r'$r_{+}$', (-np.pi / 3, rplus(self._a) - 1), color='k', fontsize=14)
            return ln,

        def update(i):
            xdata.append(self._phis[i])
            ydata.append(self._rs[i])
            xpoint.append(self._phis[i])
            ypoint.append(self._rs[i])
            if i != 0:
                xpoint.pop(0)
                ypoint.pop(0)
            if i > 100:
                xdata.pop(0)
                ydata.pop(0)
            ln.set_data(xdata, ydata)
            pt.set_data(xpoint, ypoint)
            return ln, pt,

        # noinspection PyTypeChecker
        ani = animation.FuncAnimation(fig, update,
                                      frames=np.linspace(0, len(self._ts), 500, dtype='i4', endpoint=False),
                                      init_func=init, blit=True, interval=100, repeat=False)
        ani.save('../movies/orbit_{}_{}_{}.gif'.format(self._a, self._eta, self._risco_factor),
                 fps=10, writer='imagemagick')
        plt.show()
