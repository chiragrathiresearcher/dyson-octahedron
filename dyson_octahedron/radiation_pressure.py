"""
radiation_pressure.py
======================
Solar Radiation Pressure (SRP) forces and secular orbital drift
for the octahedral Dyson swarm.

Author : Chirag Rathi
Ref    : Rathi, C. (2026). OctaDyson — arXiv preprint
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
L_SUN   = 3.828e26    # solar luminosity      [W]
C_LIGHT = 2.998e8     # speed of light        [m s⁻¹]
AU      = 1.496e11    # 1 AU                  [m]
G       = 6.674e-11
M_SUN   = 1.989e30


def srp_force(r_i: float,
              A_i: float,
              eta_i: float = 0.9,
              r_hat: np.ndarray = None) -> np.ndarray:
    """
    Solar radiation pressure force on spacecraft i (Rathi 2026, eq. 3):

        F_SRP,i = (L☉ Aᵢ)/(4π c rᵢ²) · (1 + ηᵢ) · r̂ᵢ

    Parameters
    ----------
    r_i   : float        Orbital radius [m]
    A_i   : float        Effective area [m²]
    eta_i : float        Reflectivity coefficient (0=absorb, 1=perfect reflect)
    r_hat : ndarray (3,) Unit radial vector; defaults to +x if None

    Returns
    -------
    F : ndarray (3,)  Force vector [N]
    """
    if r_hat is None:
        r_hat = np.array([1.0, 0.0, 0.0])
    r_hat = r_hat / np.linalg.norm(r_hat)
    mag   = (L_SUN * A_i) / (4 * np.pi * C_LIGHT * r_i**2) * (1 + eta_i)
    return mag * r_hat


def secular_drift(r_i: float,
                  A_i: float,
                  m_i: float,
                  t: np.ndarray,
                  eta_i: float = 0.9) -> np.ndarray:
    """
    Cumulative radial drift due to SRP (Rathi 2026, eq. 4):

        Δrᵢ(t) ≈ (3/2) · F_SRP,i / (mᵢ nᵢ²) · t

    Parameters
    ----------
    r_i   : float          Nominal orbital radius [m]
    A_i   : float          Effective area [m²]
    m_i   : float          Spacecraft mass [kg]
    t     : ndarray        Time array [s]
    eta_i : float          Reflectivity

    Returns
    -------
    delta_r : ndarray      Radial drift [m]
    """
    F_mag  = np.linalg.norm(srp_force(r_i, A_i, eta_i))
    n_i    = np.sqrt(G * M_SUN / r_i**3)        # mean motion
    return (1.5 * F_mag / (m_i * n_i**2)) * t


def delta_v_budget(r_i: float,
                   A_i: float,
                   m_i: float,
                   T_orbit: float,
                   eta_i: float = 0.9) -> float:
    """
    Annual Δv required to compensate SRP secular drift:

        Δv_year = Σᵢ ‖a_pert,i‖ · T_orbit

    Parameters
    ----------
    T_orbit : float  Orbital period [s]

    Returns
    -------
    dv : float  [m s⁻¹ yr⁻¹]
    """
    F_mag  = np.linalg.norm(srp_force(r_i, A_i, eta_i))
    a_pert = F_mag / m_i
    return a_pert * T_orbit


def srp_all_vertices(r: float = AU,
                     A_panel: float = 1e4,
                     eta: float = 0.9) -> np.ndarray:
    """
    Compute SRP force vectors for all 6 octahedral vertices.
    Demonstrates near-perfect force cancellation (±x, ±y, ±z pairs).

    Returns
    -------
    forces : ndarray (6, 3) [N]
    """
    unit_vecs = np.array([
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
    ], dtype=float)
    forces = np.array([
        srp_force(r, A_panel, eta, uv) for uv in unit_vecs
    ])
    return forces


def srp_net_force(r: float = AU,
                  A_panel: float = 1e4,
                  eta: float = 0.9) -> np.ndarray:
    """
    Net force on the swarm — should be ~0 for the octahedron.
    Quantifies the symmetry advantage over other geometries.
    """
    return srp_all_vertices(r, A_panel, eta).sum(axis=0)


def drift_vs_time(r: float       = AU,
                  A_i: float     = 1e4,
                  m_i: float     = 1e3,
                  t_years: float = 10.0,
                  n_pts: int     = 300,
                  eta_i: float   = 0.9) -> dict:
    """
    Compute drift over a multi-year span for analysis & plotting.

    Returns
    -------
    dict with 't_years', 'drift_m', 'drift_km'
    """
    t_sec   = np.linspace(0, t_years * 3.156e7, n_pts)
    drift   = secular_drift(r, A_i, m_i, t_sec, eta_i)
    AU      = 1.496e11
    # NOTE: A/m ratio of 1e4/1e3 = 10 m²/kg is solar-sail territory.
    # Realistic power satellites would have A/m ~0.01–0.1 m²/kg,
    # reducing drift by 100–1000×. Drift shown in AU for context.
    return {
        't_years'  : t_sec / 3.156e7,
        'drift_m'  : drift,
        'drift_km' : drift / 1e3,
        'drift_AU' : drift / AU,
        'a_over_m' : A_i / m_i,
    }


def luminosity_perturbation(r: float     = AU,
                             gamma: float = 0.001,  # CORRECTED: 0.1% TSI variation, not 1%
                             omega: float = 2 * np.pi / (11 * 3.156e7),
                             n_pts: int   = 500) -> dict:
    """
    Orbital perturbation due to sinusoidal stellar luminosity variation
    (Rathi 2026, Sec. 5):

        L(t) = L₀ (1 + γ sin ω t)
        δr/r ≈ γ ω² / (2 n²)

    Parameters
    ----------
    gamma : float   Fractional luminosity amplitude (solar cycle ≈ 0.1 %)
    omega : float   Angular frequency of variation [rad s⁻¹]

    Returns
    -------
    dict with 't_years', 'L_t', 'delta_r_over_r'
    """
    t      = np.linspace(0, 22 * 3.156e7, n_pts)  # two solar cycles
    L_t    = L_SUN * (1 + gamma * np.sin(omega * t))
    n_orb  = np.sqrt(G * M_SUN / r**3)
    dr_r   = gamma * omega**2 / (2 * n_orb**2)     # scalar magnitude
    return {
        't_years'       : t / 3.156e7,
        'L_t'           : L_t,
        'delta_r_over_r': dr_r * np.ones_like(t),   # constant envelope
        'L_variation'   : gamma * np.sin(omega * t),
    }
