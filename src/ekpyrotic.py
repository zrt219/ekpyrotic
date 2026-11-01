"""OD-QC-001 — Ekpyrotic contraction, anisotropy smoothing, effective bounce,
and WKB classicality.

Conventions
-----------
* Reduced Planck units: M_pl = 1, so the Friedmann constraint reads
  3 H^2 = rho.  Signature (-,+,+,+).
* Matter: canonical scalar field phi with exponential potential
      V(phi) = s V0 exp(-c phi),   c = sqrt(2/p),   s = -1 for p < 1/3
  (ekpyrotic regime, w > 1) and s = +1 for p > 1/3 (the w < 1
  counterexample).  On the scaling attractor a ∝ |t|^p (cosmic time t < 0,
  contraction) and w = -1 + 2/(3p).
* Anisotropy is tracked as a shear energy density rho_Sigma (state component
  R) obeying d rho_Sigma/dt = -6 H rho_Sigma  (rho_Sigma ∝ a^-6): the exact
  behaviour of the free-gravitational-field shear of a Bianchi I spacetime
  sourced by an isotropic fluid.
* Therefore  rho_Sigma/rho_phi ∝ a^-6 / a^-3(1+w) = a^3(w-1).

Integration variables
---------------------
* GR contraction: integrated in contraction e-folds  n = -ln(a/a_i),
  dn/dt = h = -H > 0.  The attractor is exactly exponential in n, which
  makes the integration uniformly accurate all the way to the crunch
  (in cosmic time, the integrator eventually drifts onto a time-shifted
  attractor).  State: y = (phi, u, h, R, t) with u = dphi/dt and t carried
  along via dt/dn = 1/h.  alpha = ln a = -n (a_i = 1).
* Effective bounce: integrated in cosmic time t (the bounce spans only a
  modest density range).  State: y = (phi, u, H, R, alpha).
* In both systems H (or h) is evolved by a (modified) Raychaudhuri equation,
  so the (modified) Friedmann constraint is an independent residual for
  accuracy control.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Potential and analytic scaling solution
# ---------------------------------------------------------------------------


def potential(p, V0=1.0):
    """Return (V, dV, c) for the exponential scaling potential, c = sqrt(2/p).

    V(phi) = -V0 exp(-c phi)  for p < 1/3  (ekpyrotic regime, w > 1),
    V(phi) = +V0 exp(-c phi)  for p > 1/3  (w < 1 counterexample).
    The power-law scaling solution a ∝ |t|^p requires V to have the sign of
    (3p - 1); the potential sign is chosen accordingly.
    """
    c = np.sqrt(2.0 / p)
    sgn = -1.0 if p < 1.0 / 3.0 else 1.0

    def V(phi):
        return sgn * V0 * np.exp(-c * np.asarray(phi, dtype=float))

    def dV(phi):
        return -sgn * c * V0 * np.exp(-c * np.asarray(phi, dtype=float))

    return V, dV, c


def w_of_p(p):
    """Equation of state of the scaling attractor."""
    return -1.0 + 2.0 / (3.0 * p)


def shear_ratio_exponent(p):
    """Exponent m in rho_Sigma/rho_phi ∝ a^m on the attractor: m = 3(w-1)."""
    return 3.0 * (w_of_p(p) - 1.0)  # = 2/p - 6


def classicality_exponent(p):
    """Analytic attractor prediction: Q ∝ a^((1-3p)/p) as the crunch nears."""
    return (1.0 - 3.0 * p) / p


def attractor_t(p, V0, t, t_i):
    """Exact scaling solution at cosmic time t < 0, with a(t_i) = 1, t_i < 0.

    a = (|t|/|t_i|)^p,  |V| = p|1-3p|/t^2,  u = dphi/dt = 2/(c t),  H = p/t.
    """
    c = np.sqrt(2.0 / p)
    alpha = p * np.log(abs(t) / abs(t_i))
    phi = ((2.0 / c) * np.log(abs(t))
           - (1.0 / c) * np.log(p * abs(1 - 3 * p) / V0))
    u = 2.0 / (c * t)
    H = p / t
    return alpha, phi, u, H


def attractor_n(p, V0, n, t_i):
    """Exact scaling solution in e-fold time: a(t_i) = 1, n = 0 at t_i.

    t = t_i e^{-n/p},  phi = phi_i - (2/(c p)) n,  u = u_i e^{n/p},
    h = h_i e^{n/p},  R = R_i e^{6 n}.
    """
    _, phi_i, u_i, H_i = attractor_t(p, V0, t_i, t_i)
    c = np.sqrt(2.0 / p)
    return dict(
        t=t_i * np.exp(-n / p),
        phi=phi_i - 2.0 / (c * p) * n,
        u=u_i * np.exp(n / p),
        h=-H_i * np.exp(n / p),
        alpha=-n,
    )


# ---------------------------------------------------------------------------
# GR contraction in e-fold time  n = -ln a
# ---------------------------------------------------------------------------


def initial_state(p, V0, t_i, shear_frac,
                  du_rel=0.0, dphi=0.0, dH_rel=0.0, dt=0.0, on_constraint=True):
    """Attractor initial data with optional homogeneous perturbations.

    Returns the n-time state y0 = (phi, u, h, R, t).  Perturbations:
    du_rel scales u, dphi shifts phi, dH_rel scales H, dt shifts the time
    label (the marginal mode).  With on_constraint=True, H is rescaled so
    3 h^2 = rho_tot exactly; with False the GR constraint residual is
    nonzero at n=0 by construction.
    """
    V, dV, c = potential(p, V0)
    _, phi_i, u_i, H_i = attractor_t(p, V0, t_i, t_i)
    phi_i, u_i, H_i, t_i = phi_i + dphi, u_i * (1 + du_rel), H_i * (1 + dH_rel), t_i + dt
    R_i = shear_frac * 3.0 * H_i**2
    h_i = -H_i
    if on_constraint:
        h_i = -np.sign(H_i) * np.sqrt((0.5 * u_i**2 + V(phi_i) + R_i) / 3.0)
    return np.array([phi_i, u_i, h_i, R_i, t_i])


def integrate_gr(p, V0, t_i, shear_frac, rho_stop=1e10,
                 rtol=1e-11, atol=1e-13, max_step=np.inf, y0=None):
    """Contracting GR solution, integrated in n, a(t_i) = 1 (n = 0 at t_i).

    y0 overrides the default (attractor) initial state (see initial_state).
    rho_Sigma(t_i)/rho_phi(t_i) = shear_frac.  Stops at a terminal event
    rho_tot = rho_stop: a trust cutoff on the classical integration, not a
    property of the solution (classically rho → ∞ at finite t).

    Returns (sol, pot): sol.t is n, sol.sol(n) gives (phi, u, h, R, t).
    """
    V, dV, c = potential(p, V0)
    if y0 is None:
        y0 = initial_state(p, V0, t_i, shear_frac)

    def rhs(n, y):
        phi, u, h, R, t = y
        return [
            u / h,                       # dphi/dn
            3.0 * u - dV(phi) / h,       # du/dn
            (0.5 * u * u + R) / h,       # dh/dn  (Raychaudhuri)
            6.0 * R,                     # dR/dn
            1.0 / h,                     # dt/dn
        ]

    def ev_stop(n, y):
        return 0.5 * y[1] ** 2 + V(y[0]) + y[3] - rho_stop

    ev_stop.terminal = True
    ev_stop.direction = 1

    sol = solve_ivp(rhs, (0.0, 1e6), y0, method="DOP853",
                    rtol=rtol, atol=atol, dense_output=True,
                    events=[ev_stop], max_step=max_step)
    return sol, potential(p, V0)


# ---------------------------------------------------------------------------
# Effective bounce in cosmic time
# ---------------------------------------------------------------------------


def integrate_effective(p, V0, t_i, shear_frac, rho_c=0.41,
                        rtol=1e-11, atol=1e-13, t_end=None,
                        stop_at_rho0=True, v_flip=False, floor_factor=None):
    """Effective bounce model in cosmic time, a(t_i) = 1.

    Modified Friedmann equation  H^2 = (rho/3)(1 - rho/rho_c)  with the
    continuity equation unchanged, giving the consistent modified
    Raychaudhuri equation  dot H = -(rho+p)/2 (1 - 2 rho/rho_c).  This
    dynamics is *specified*, not derived here — see the report.

    stop_at_rho0=True (default): terminal event at rho_tot = 0 — with
    V < 0 the kinetic energy redshifts in the expanding branch and the
    universe recollapses; the run covers exactly one
    contraction-bounce-expansion window.

    stop_at_rho0=False: the run continues through the recollapses, recording
    every H = 0 crossing (bounces and recollapses) in sol.t_events[0] —
    used to study the bounce-accumulation cascade.

    v_flip=True: the sign of V (and V_,phi) follows the sign of H, i.e. the
    potential is negative while contracting and positive while expanding —
    a crude stand-in for the potential 'turning positive' at the bounce.
    """
    V0c = np.sqrt(2.0 / p)
    s_pot = -1.0 if p < 1.0 / 3.0 else 1.0  # potential sign by regime

    def Vf(phi, H):
        v = s_pot * V0 * np.exp(-V0c * np.asarray(phi, dtype=float))
        if v_flip:
            v = np.sign(H) * v
        return v

    def dVf(phi, H):
        d = -s_pot * V0c * V0 * np.exp(-V0c * np.asarray(phi, dtype=float))
        if v_flip:
            d = np.sign(H) * d
        return d

    alpha_i, phi_i, u_i, H_i = attractor_t(p, V0, t_i, t_i)
    R_i = shear_frac * 3.0 * H_i**2
    # put the ICs exactly on the constraint surface: 3 H_i^2 = rho_phi + R
    H_i = H_i * np.sqrt(1.0 + shear_frac)
    y0 = [phi_i, u_i, H_i, R_i, alpha_i]

    def rhs(t, y):
        phi, u, H, R, alpha = y
        rho = 0.5 * u * u + Vf(phi, H) + R
        return [
            u,
            -3.0 * H * u - dVf(phi, H),
            -(0.5 * u * u + R) * (1.0 - 2.0 * rho / rho_c),
            -6.0 * H * R,
            H,
        ]

    def ev_bounce(t, y):
        return y[2]

    ev_bounce.terminal = False
    ev_bounce.direction = 1

    events = [ev_bounce]
    if stop_at_rho0:
        def ev_floor(t, y):
            return 0.5 * y[1] ** 2 + Vf(y[0], y[2]) + y[3]

        ev_floor.terminal = True
        ev_floor.direction = -1
        events.append(ev_floor)
    elif floor_factor is not None:
        # stop near (but not at) the rho -> 0 bottleneck: lets the run record
        # many bounce cycles without grinding into the accumulation point
        def ev_floor2(t, y):
            return 0.5 * y[1] ** 2 + Vf(y[0], y[2]) + y[3] - floor_factor * rho_c

        ev_floor2.terminal = True
        ev_floor2.direction = -1
        events.append(ev_floor2)

    t1 = abs(t_i) if t_end is None else t_end
    sol = solve_ivp(rhs, (t_i, t1), y0, method="DOP853",
                    rtol=rtol, atol=atol, dense_output=True,
                    events=events)
    return sol, potential(p, V0)


# ---------------------------------------------------------------------------
# Constraints and diagnostics
# ---------------------------------------------------------------------------


def rho_tot(pot, phi, u, R):
    V, _, _ = pot
    return 0.5 * u * u + V(phi) + R


def constraint_residual_gr(pot, y):
    """C = 3 h^2 - rho_tot for the n-time GR system (must vanish)."""
    phi, u, h, R, t = y
    return 3.0 * h * h - rho_tot(pot, phi, u, R)


def constraint_residual_eff(pot, y, rho_c):
    """C_eff = 3 H^2 - rho (1 - rho/rho_c) for the effective model."""
    phi, u, H, R, alpha = y
    rho = rho_tot(pot, phi, u, R)
    return 3.0 * H * H - rho * (1.0 - rho / rho_c)


def fit_exponent(a, ratio, a_max=None):
    """Fit ratio ∝ a^m (log-log least squares), optionally on a <= a_max.

    Returns (m, R^2)."""
    a = np.asarray(a, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    mask = np.isfinite(ratio) & (ratio > 0) & (a > 0)
    if a_max is not None:
        mask &= a <= a_max
    if mask.sum() < 3:
        return np.nan, np.nan
    x = np.log(a[mask])
    yv = np.log(ratio[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), *_ = np.linalg.lstsq(A, yv, rcond=None)
    pred = m * x + b
    ss = np.sum((yv - yv.mean()) ** 2)
    r2 = 1.0 - np.sum((yv - pred) ** 2) / ss if ss > 0 else np.nan
    return m, r2


# ---------------------------------------------------------------------------
# WKB classicality along a classical trajectory
# ---------------------------------------------------------------------------


def classicality_Q(pot, alpha, phi, u, H, hbar=1.0):
    """WKB classicality quantity Q = hbar |nabla^2 S| / |(grad S)^2|

    for the minisuperspace Wheeler-DeWitt equation (factor ordering chosen so
    that no first derivatives appear)

        [ hbar^2 (d^2/dalpha^2 - 6 d^2/dphi^2) + 12 a^6 V(phi) ] Psi = 0,

    whose Hamilton-Jacobi equation is
        -(S_,alpha)^2 + 6 (S_,phi)^2 + 12 a^6 V = 0.

    On the classical flow the momenta are S_,alpha = -6 a^3 H and
    S_,phi = a^3 u, so

        S_,alpha,alpha = -18 a^3 H,
        S_,phi,phi     = a^3 (du/dt) / u,   du/dt = -3 H u - V_,phi.

    All quantities are evaluated from the numerical trajectory (H is the
    *signed* Hubble rate); no analytic attractor scalings are assumed inside
    this function.  Q << 1: WKB amplitude slowly varying compared with the
    phase — the wavefunction is peaked on a classical history.
    """
    V, dV, _ = pot
    alpha = np.asarray(alpha, float)
    a3 = np.exp(3.0 * alpha)
    du_dt = -3.0 * H * u - dV(phi)

    P_alpha = -6.0 * a3 * H
    P_phi = a3 * u
    grad_S_sq = np.abs(-(P_alpha**2) + 6.0 * P_phi**2)  # = 12 a^6 |V|

    lap_S = -18.0 * a3 * H - 6.0 * a3 * (du_dt / u)

    return hbar * np.abs(lap_S) / grad_S_sq
