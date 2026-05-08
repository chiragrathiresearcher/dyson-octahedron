"""
station_keeping.py — OctaDyson v2.0
=====================================
Delta-v budget, propellant mass, and formation-keeping analysis
for the octahedral Dyson swarm.

Physics:
  SRP secular drift (Gauss variational equations):
      da/dt = (2 F_SRP)/(n m)   [semi-major axis drift rate]
  Annual station-keeping delta-v:
      dv_yr = (F_SRP / m) * T_year
  LQR-based optimal formation control cost estimate.
"""
import numpy as np

AU      = 1.496e11
G       = 6.674e-11
M_SUN   = 1.989e30
L_SUN   = 3.828e26
C_LIGHT = 2.998e8
T_YEAR  = 3.156e7   # seconds


def mean_motion(r):
    return np.sqrt(G * M_SUN / r**3)


def srp_force(r, A_panel, eta_ref=0.9):
    """SRP force magnitude on one spacecraft [N]."""
    return (L_SUN * A_panel * (1 + eta_ref)) / (4 * np.pi * C_LIGHT * r**2)


def drift_rate(r, A_panel, mass, eta_ref=0.9):
    """
    Semi-major axis drift rate from SRP [m/s² effectively m/s per s].
    From Gauss variational equations, radial SRP:
        da/dt = 2 * F_SRP / (n * m)
    """
    F = srp_force(r, A_panel, eta_ref)
    n = mean_motion(r)
    return 2 * F / (n * mass)   # m/s


def annual_deltav(r, A_panel, mass, eta_ref=0.9):
    """
    Annual station-keeping delta-v per spacecraft [m/s/year].
    Conservative estimate: continuous thrust to cancel SRP acceleration.
        dv = (F_SRP / m) * T_year
    """
    F  = srp_force(r, A_panel, eta_ref)
    return (F / mass) * T_YEAR


def propellant_mass(dv_yr, mass_dry, Isp=3000.0, years=10.0):
    """
    Tsiolkovsky propellant mass for station-keeping over mission lifetime.
    Isp = 3000 s corresponds to high-efficiency ion/Hall-effect thruster.
    """
    g0    = 9.807
    dv_tot = dv_yr * years
    m_prop = mass_dry * (np.exp(dv_tot / (Isp * g0)) - 1)
    return m_prop


def deltav_vs_radius(r_min=0.3*AU, r_max=2.0*AU, n_pts=200,
                     A_panel=1e4, mass=1e3):
    radii = np.linspace(r_min, r_max, n_pts)
    dv    = np.array([annual_deltav(r, A_panel, mass) for r in radii])
    dr    = np.array([drift_rate(r, A_panel, mass) for r in radii])
    return {'r_AU': radii/AU, 'dv_mps_yr': dv, 'drift_mps': dr}


def lqr_control_cost(r, A_panel, mass, tolerance_m=1e3):
    """
    Rough LQR-based optimal control delta-v estimate.
    Assumes periodic correction every T_corr = sqrt(2*tol/a_srp).
    """
    a_srp   = srp_force(r, A_panel, mass) / mass
    T_corr  = np.sqrt(2 * tolerance_m / a_srp)
    dv_corr = a_srp * T_corr   # velocity impulse per correction
    n_corr  = T_YEAR / T_corr  # corrections per year
    return {'T_corr_days': T_corr/86400,
            'n_corr_yr'  : n_corr,
            'dv_per_corr': dv_corr,
            'dv_annual'  : dv_corr * n_corr}


def swarm_fuel_summary(r=AU, A_panel=1e4, mass=1e3,
                       Isp=3000.0, years=10.0):
    dv  = annual_deltav(r, A_panel, mass)
    mp  = propellant_mass(dv, mass, Isp, years)
    lqr = lqr_control_cost(r, A_panel, mass)
    return {
        'dv_annual_mps'       : dv,
        'dv_total_mps'        : dv * years,
        'propellant_kg'       : mp,
        'propellant_fraction' : mp / (mass + mp),
        'Isp_s'               : Isp,
        'mission_years'       : years,
        'lqr_T_corr_days'     : lqr['T_corr_days'],
        'lqr_dv_annual'       : lqr['dv_annual'],
        'srp_force_N'         : srp_force(r, A_panel),
        'A_over_m'            : A_panel / mass,
    }
