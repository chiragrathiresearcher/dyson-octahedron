"""
coverage_analysis.py — OctaDyson v2.0
=======================================
Sky coverage, beam footprint, and power delivery map
for the octahedral Dyson swarm.

Key analyses:
  1. Angular coverage: fraction of 4pi steradians accessible to swarm
  2. Power delivery map: eta_total over the celestial sphere
  3. Dead-zone analysis: sky directions with no direct line-of-sight relay
  4. Multi-source illumination: how many S/C can reach any sky point
"""
import numpy as np

AU    = 1.496e11
ALPHA = 1e-15
BETA  = 0.05
R_B   = 0.5
ETA0  = 0.95

# Octahedron vertex directions (unit vectors)
VERTICES = np.array([
    [ 1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
], dtype=float)
SC_LABELS = ['+x','−x','+y','−y','+z','−z']
ANTIPODAL = {0:1,1:0,2:3,3:2,4:5,5:4}


def beam_eta(D_AU):
    """Single-hop efficiency."""
    return ETA0 * np.exp(-ALPHA*D_AU - BETA*(D_AU/R_B)**2)


def _sphere_grid(n_lat=180, n_lon=360):
    """Uniform grid over the unit sphere."""
    lat  = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon  = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)
    x = np.cos(LAT)*np.cos(LON)
    y = np.cos(LAT)*np.sin(LON)
    z = np.sin(LAT)
    return LAT, LON, np.stack([x,y,z], axis=-1)


def power_delivery_map(r_AU=1.0, n_lat=120, n_lon=240):
    """
    For each point on the celestial sphere, compute the best single-hop
    eta_beam from any spacecraft.  Returns a coverage map.
    """
    LAT, LON, dirs = _sphere_grid(n_lat, n_lon)  # dirs: (n_lat,n_lon,3)
    r = r_AU  # AU units

    # Distance from each S/C vertex to each sky direction point
    # A sky point at direction d is at distance |r*d - r*v_i| = r*|d - v_i|
    eta_map  = np.zeros((n_lat, n_lon))
    best_sc  = np.zeros((n_lat, n_lon), dtype=int)
    n_vis    = np.zeros((n_lat, n_lon), dtype=int)

    for i, v in enumerate(VERTICES):
        # Angular separation -> chord distance
        cos_sep  = np.clip(np.einsum('ijk,k->ij', dirs, v), -1, 1)
        chord    = r * np.sqrt(2*(1 - cos_sep))   # in AU
        eta_hop  = beam_eta(chord)
        mask     = eta_hop > eta_map
        eta_map  = np.where(mask, eta_hop, eta_map)
        best_sc  = np.where(mask, i, best_sc)
        n_vis   += (eta_hop > 0.1).astype(int)

    return {
        'LAT'    : LAT,
        'LON'    : LON,
        'eta_map': eta_map,
        'best_sc': best_sc,
        'n_vis'  : n_vis,
        'coverage_frac': float((eta_map > 0.3).mean()),
        'mean_eta': float(eta_map.mean()),
        'min_eta' : float(eta_map.min()),
    }


def multi_source_coverage(r_AU=1.0):
    """
    For each sky direction, compute redundancy: how many S/C have
    direct line-of-sight (defined as eta > 10%).
    """
    _, _, dirs = _sphere_grid(90, 180)
    counts = np.zeros(dirs.shape[:2], dtype=int)
    for v in VERTICES:
        cos_sep = np.einsum('ijk,k->ij', dirs, v)
        chord   = r_AU * np.sqrt(2*(1-np.clip(cos_sep,-1,1)))
        counts += (beam_eta(chord) > 0.1).astype(int)
    vals, cnt = np.unique(counts, return_counts=True)
    total = counts.size
    return {k: v/total for k,v in zip(vals.tolist(), cnt.tolist())}


def angular_coverage_stats(r_AU=1.0):
    """Summary statistics for coverage quality."""
    result = power_delivery_map(r_AU=r_AU)
    eta = result['eta_map']
    return {
        'r_AU'           : r_AU,
        'mean_eta'       : eta.mean(),
        'median_eta'     : np.median(eta),
        'min_eta'        : eta.min(),
        'frac_above_50pct': (eta > 0.5).mean(),
        'frac_above_30pct': (eta > 0.3).mean(),
        'frac_above_10pct': (eta > 0.1).mean(),
        'redundancy'     : multi_source_coverage(r_AU),
    }


def coverage_vs_radius(radii_AU=None):
    if radii_AU is None:
        radii_AU = np.linspace(0.5, 2.0, 10)
    stats = [angular_coverage_stats(r) for r in radii_AU]
    return {
        'r_AU'            : radii_AU,
        'mean_eta'        : np.array([s['mean_eta'] for s in stats]),
        'frac_above_50pct': np.array([s['frac_above_50pct'] for s in stats]),
        'min_eta'         : np.array([s['min_eta'] for s in stats]),
    }
