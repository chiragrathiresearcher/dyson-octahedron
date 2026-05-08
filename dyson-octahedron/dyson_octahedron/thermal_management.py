"""
thermal_management.py  [OctaDyson v1.1.0 — CORRECTED]
======================
Waste-heat generation and radiator sizing for the octahedral Dyson swarm.

BUG FIX (v1.1.0): Original code computed P_waste = P_harvest*(1-eta_pv),
where P_harvest already had eta_pv applied. Underestimated waste heat by
factor (1-eta_pv)/eta_pv (~2.33x at eta_pv=30%).

Correct derivation:
    P_intercept = (L☉/4πr²) × A          [total intercepted]
    P_harvest   = P_intercept × eta_pv    [electrical output]
    P_waste     = P_intercept × (1-eta_pv) = P_harvest*(1-eta_pv)/eta_pv
"""
import numpy as np

SIGMA   = 5.6704e-8
T_SPACE = 3.0
AU      = 1.496e11
L_SUN   = 3.828e26


def intercepted_power(r, A_panel):
    """Total solar power striking the panel — before conversion losses."""
    return (L_SUN / (4 * np.pi * r**2)) * A_panel


def harvested_power(r, A_panel, eta_pv=0.30):
    """Net electrical power: P_intercept × eta_pv."""
    return intercepted_power(r, A_panel) * eta_pv


def waste_heat(r, A_panel, eta_pv=0.30):
    """
    CORRECTED: P_waste = P_intercept × (1-eta_pv).
    Equivalently: P_harvest × (1-eta_pv)/eta_pv.
    """
    return intercepted_power(r, A_panel) * (1.0 - eta_pv)


def radiator_area(P_waste, T_rad, epsilon=0.90):
    """A_rad = P_waste / [σ ε (T_rad⁴ - T_space⁴)]"""
    return P_waste / (SIGMA * epsilon * (T_rad**4 - T_SPACE**4))


def equilibrium_temperature(P_waste, A_rad, epsilon=0.90):
    """T_rad = [(P_waste/(σ ε A_rad)) + T_space⁴]^(1/4)"""
    return ((P_waste / (SIGMA * epsilon * A_rad)) + T_SPACE**4) ** 0.25


def thermal_profile_vs_radius(r_values, A_panel=1e4, T_rad=400.0,
                               eta_pv=0.30, epsilon=0.90):
    P_h = np.array([harvested_power(r, A_panel, eta_pv) for r in r_values])
    P_w = np.array([waste_heat(r, A_panel, eta_pv)      for r in r_values])
    A_r = np.array([radiator_area(pw, T_rad, epsilon)   for pw in P_w])
    T_eq = np.array([equilibrium_temperature(pw, A_panel*0.5, epsilon) for pw in P_w])
    return {'r_AU': r_values/AU, 'P_harvest': P_h, 'P_waste': P_w,
            'A_rad': A_r, 'T_eq': T_eq}


def swarm_thermal_summary(r=AU, A_panel=1e4, T_rad=400.0,
                           eta_pv=0.30, epsilon=0.90):
    Ph  = harvested_power(r, A_panel, eta_pv)
    Pw  = waste_heat(r, A_panel, eta_pv)
    Ar  = radiator_area(Pw, T_rad, epsilon)
    Teq = equilibrium_temperature(Pw, Ar, epsilon)
    return {
        'n_spacecraft': 6,
        'P_intercept_W'     : intercepted_power(r, A_panel),
        'P_harvest_each_W'  : Ph,
        'P_waste_each_W'    : Pw,
        'A_rad_each_m2'     : Ar,
        'T_equilibrium_K'   : Teq,
        'P_harvest_total_W' : 6*Ph,
        'P_waste_total_W'   : 6*Pw,
        'A_rad_total_m2'    : 6*Ar,
        'rad_to_panel_ratio': Ar/A_panel,
        'eta_pv'            : eta_pv,
    }


def eta_pv_sensitivity(eta_range, r=AU, A_panel=1e4, T_rad=400.0, epsilon=0.90):
    P_h = np.array([harvested_power(r, A_panel, e) for e in eta_range])
    P_w = np.array([waste_heat(r, A_panel, e)      for e in eta_range])
    A_r = np.array([radiator_area(pw, T_rad, epsilon) for pw in P_w])
    return {'eta_pv': eta_range, 'P_harvest': P_h, 'P_waste': P_w, 'A_rad': A_r}
