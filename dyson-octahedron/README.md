# OctaDyson v2.0

A complete Python simulation framework for octahedral Dyson swarm architectures.

## Modules

| Module | Description |
|--------|-------------|
| orbital_dynamics.py | CW dynamics, Lyapunov stability, perturbation simulation |
| radiation_pressure.py | SRP force, cancellation, secular drift |
| thermal_management.py | Waste heat, radiator sizing, sensitivity |
| beam_efficiency.py | CSBPB relay, geometry comparison, stochastic chain |
| station_keeping.py | ΔV budget, propellant mass, LQR control |
| coverage_analysis.py | Spherical sky coverage, power delivery map |
| mass_budget.py | Spacecraft mass breakdown, TRL assessment |

## Quick Start

```bash
pip install -r requirements.txt
python main.py --radius 1.0 --area 1e4 --mass 1e3 --eta-pv 0.30
```

## Key Results (r=1 AU, A=10⁴ m², η_PV=30%)

- Lyapunov exponent: λ_max = 0 (neutrally stable)
- SRP net force: 0 N (exact antipodal cancellation)
- Radiator area: ~7,300 m² per spacecraft (ratio 0.73 to panel)
- Annual ΔV: ~2.7 m/s/yr (ion propulsion required)
- Sky coverage: ~63% mean η_beam, ~68% above 30% threshold
- Beam efficiency (oct. edge): 63.7%

## Licence
MIT — github.com/chiragrathiresearcher/dyson-octahedron
