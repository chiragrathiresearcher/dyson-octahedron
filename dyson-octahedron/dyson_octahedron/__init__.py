"""
OctaDyson
=========
Quantitative simulation framework for octahedral Dyson swarm architectures.

Author  : Chirag Rathi
Version : 1.0.0
License : MIT
arXiv   : astro-ph.SR (submitted 2026)

Modules
-------
orbital_dynamics    — 3D orbital simulation & Lyapunov stability
radiation_pressure  — SRP forces & secular drift
thermal_management  — Waste heat & radiator sizing
beam_efficiency     — CSBPB relay efficiency
visualization       — Publication-quality plots
"""

__version__  = "1.0.0"
__author__   = "Chirag Rathi"
__email__    = "chiragrathiresearcher@github"
__license__  = "MIT"

from dyson_octahedron.orbital_dynamics    import (
    octahedral_positions, mean_motion, cw_matrix,
    lyapunov_spectrum, simulate_perturbations,
    simulate_symmetry_control,
)
from dyson_octahedron.radiation_pressure  import (
    srp_force, secular_drift, srp_all_vertices,
    srp_net_force, drift_vs_time, luminosity_perturbation,
)
from dyson_octahedron.thermal_management  import (
    harvested_power, waste_heat, radiator_area,
    swarm_thermal_summary, thermal_profile_vs_radius,
    eta_pv_sensitivity,
)
from dyson_octahedron.beam_efficiency     import (
    beam_attenuation, relay_efficiency,
    efficiency_vs_distance, geometry_comparison,
    multi_hop_vs_single, relay_chain_simulation,
)

__all__ = [
    "octahedral_positions", "mean_motion", "cw_matrix",
    "lyapunov_spectrum", "simulate_perturbations",
    "simulate_symmetry_control",
    "srp_force", "secular_drift", "srp_all_vertices",
    "srp_net_force", "drift_vs_time", "luminosity_perturbation",
    "harvested_power", "waste_heat", "radiator_area",
    "swarm_thermal_summary", "thermal_profile_vs_radius",
    "eta_pv_sensitivity",
    "beam_attenuation", "relay_efficiency",
    "efficiency_vs_distance", "geometry_comparison",
    "multi_hop_vs_single", "relay_chain_simulation",
]
