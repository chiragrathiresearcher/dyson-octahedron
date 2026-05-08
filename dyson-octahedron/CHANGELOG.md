# OctaDyson — Changelog

## v2.0.0 (2026-05) — MAJOR RELEASE

### New Modules
- **station_keeping.py**: Delta-v budget, propellant mass (Tsiolkovsky),
  LQR control correction intervals, Isp sensitivity analysis.
- **coverage_analysis.py**: Full spherical sky coverage map, per-direction
  beam efficiency, multi-source redundancy, coverage vs orbital radius.
- **mass_budget.py**: Complete spacecraft mass breakdown (PV, radiators,
  structure, avionics, propellant), specific power, TRL assessment.

### New Figures (9 total, up from 7)
- fig3_station_keeping: ΔV vs radius, propellant fraction vs Isp,
  mass waterfall, LQR correction frequency.
- fig7_coverage: Full-sky power delivery map, redundancy histogram,
  coverage vs radius, per-spacecraft sky fraction.
- fig8_mass_trl: Mass donut, specific power vs area, TRL bars,
  mass sensitivity, mission lifetime propellant.
- fig9_dashboard: Complete 12-panel system dashboard.

### Physics Fixes (carried from v1.1)
- Waste heat formula: P_waste = P_intercept × (1−η_PV)
- Lyapunov figure: eigenfrequency spectrum replaces noise plot
- Multi-hop relay: actual vertex paths only
- Solar cycle gamma: 0.001 (0.1%)

## v1.1.0 (2026-05) — Bug fix release
## v1.0.0 (2026-02) — Initial release
