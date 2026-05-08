"""
mass_budget.py — OctaDyson v2.0
=================================
Complete spacecraft mass breakdown and technology readiness analysis.

Sub-system mass fractions are informed by published solar-sail, space
solar power, and large deployable satellite literature.
"""
import numpy as np

AU     = 1.496e11
L_SUN  = 3.828e26
SIGMA  = 5.6704e-8


def pv_panel_mass(A_panel, areal_density_kg_m2=0.5):
    """PV panel mass. State-of-art thin-film: ~0.5 kg/m². Future target: 0.1 kg/m²."""
    return A_panel * areal_density_kg_m2


def radiator_mass(A_rad, areal_density_kg_m2=2.0):
    """Deployable radiator mass. Carbon-fibre loop-heat-pipe: ~2 kg/m²."""
    return A_rad * areal_density_kg_m2


def structural_mass(m_pv, m_rad, struct_frac=0.15):
    """Structural mass as fraction of payload (panel + radiator)."""
    return struct_frac * (m_pv + m_rad)


def propellant_mass_tsiol(dv_total, m_dry, Isp=3000.0):
    return m_dry * (np.exp(dv_total / (Isp * 9.807)) - 1)


def avionics_mass(A_panel, avionics_base=20.0):
    """Avionics, comms, ADCS. Scales weakly with panel area."""
    return avionics_base + 0.001 * A_panel


def full_mass_budget(r=AU, A_panel=1e4, Isp=3000.0,
                     years=10.0, eta_pv=0.30,
                     pv_rho=0.5, rad_rho=2.0):
    """Complete spacecraft mass breakdown."""
    from dyson_octahedron.thermal_management import waste_heat, radiator_area
    from dyson_octahedron.station_keeping    import annual_deltav

    A_rad   = radiator_area(waste_heat(r, A_panel, eta_pv), 400.0)
    m_pv    = pv_panel_mass(A_panel, pv_rho)
    m_rad   = radiator_mass(A_rad, rad_rho)
    m_struct= structural_mass(m_pv, m_rad)
    m_av    = avionics_mass(A_panel)
    m_dry   = m_pv + m_rad + m_struct + m_av

    dv_yr   = annual_deltav(r, A_panel, m_dry)
    dv_tot  = dv_yr * years
    m_prop  = propellant_mass_tsiol(dv_tot, m_dry, Isp)
    m_wet   = m_dry + m_prop

    # Power-to-mass
    P_harv  = (L_SUN/(4*np.pi*r**2)) * A_panel * eta_pv
    sp_power= P_harv / m_wet  # W/kg

    return {
        'm_pv_kg'       : m_pv,
        'm_radiator_kg' : m_rad,
        'm_structure_kg': m_struct,
        'm_avionics_kg' : m_av,
        'm_dry_kg'      : m_dry,
        'm_propellant_kg': m_prop,
        'm_wet_kg'      : m_wet,
        'prop_fraction' : m_prop / m_wet,
        'P_harvest_W'   : P_harv,
        'specific_power_W_kg': sp_power,
        'A_rad_m2'      : A_rad,
        'dv_annual_mps' : dv_yr,
        'dv_total_mps'  : dv_tot,
    }


def mass_sensitivity(param='pv_rho', values=None, r=AU, A_panel=1e4):
    """Sweep a single parameter and return mass breakdown array."""
    if values is None:
        if param == 'pv_rho':   values = np.linspace(0.05, 2.0, 30)
        elif param == 'eta_pv': values = np.linspace(0.10, 0.60, 30)
        elif param == 'A_panel':values = np.logspace(2, 5, 30)
    results = []
    for v in values:
        kw = {'r': r, 'A_panel': A_panel}
        kw[param] = v
        results.append(full_mass_budget(**kw))
    return {'values': values, 'results': results, 'param': param}


def technology_readiness():
    """
    TRL estimates for each OctaDyson sub-system, with mass/cost targets.
    Returns structured data for a radar/bar chart.
    """
    return {
        'subsystems': [
            'Thin-film PV (>30%)',
            'Deployable radiators',
            'Ion propulsion (Isp>3000s)',
            'Coherent laser relay',
            'Formation ADCS',
            'Autonomous GNC',
            'In-space assembly',
        ],
        'TRL_current': [5, 6, 7, 4, 7, 6, 3],
        'TRL_needed' : [8, 8, 8, 8, 8, 8, 8],
        'mass_fraction': [0.45, 0.35, 0.08, 0.04, 0.04, 0.02, 0.02],
        'cost_driver'  : [True, True, False, True, False, False, True],
    }
