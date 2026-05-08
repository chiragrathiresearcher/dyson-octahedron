"""
main.py  [OctaDyson v1.1.0 — CORRECTED]
=======
Master simulation runner with all bug fixes applied.

Fixes vs v1.0.0:
  1. Thermal: waste heat formula corrected (×3.33 larger)
  2. Lyapunov: noise-plot replaced with eigenfrequency spectrum
  3. Multi-hop: actual vertex paths only
  4. Gamma: luminosity perturbation default fixed 0.01→0.001
"""
import argparse, os, sys, time
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description='OctaDyson v1.1 — Corrected Simulation Suite')
    p.add_argument('--radius',   type=float, default=1.0)
    p.add_argument('--outdir',   type=str,   default='outputs_v11')
    p.add_argument('--days',     type=float, default=365.25)
    p.add_argument('--area',     type=float, default=1e4)
    p.add_argument('--mass',     type=float, default=1e3)
    p.add_argument('--eta-pv',   type=float, default=0.30)
    p.add_argument('--no-plots', action='store_true')
    return p.parse_args()

def header(t): print('\n'+'='*60+f'\n  {t}\n'+'='*60)
def row(k,v):  print(f'    {k:<30s} {v}')

def main():
    args   = parse_args()
    AU     = 1.496e11
    r      = args.radius * AU
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(__file__))
    t0 = time.time()

    from dyson_octahedron.orbital_dynamics   import (
        cw_eigenstructure, lyapunov_vs_radius,
        simulate_perturbations, simulate_symmetry_control)
    from dyson_octahedron.radiation_pressure import (
        srp_all_vertices, srp_net_force, drift_vs_time, luminosity_perturbation)
    from dyson_octahedron.thermal_management import swarm_thermal_summary, \
        thermal_profile_vs_radius, eta_pv_sensitivity
    from dyson_octahedron.beam_efficiency    import (
        efficiency_vs_distance, geometry_comparison,
        multi_hop_vs_single, relay_chain_simulation)

    # 1. Orbital
    header('1. ORBITAL DYNAMICS [CORRECTED]')
    eig_info = cw_eigenstructure(r)
    row('λ_max (analytical)', f'{eig_info["lambda_max"]:.4e} s⁻¹ (neutral)')
    row('Noise floor |Re(λ)|/n', f'{eig_info["noise_floor"]:.2e} (~machine ε)')
    row('Stability', eig_info['stability'].upper() + ' — active control required')

    radii, freq1, freq2 = lyapunov_vs_radius()
    sim_result  = simulate_perturbations(r=r, t_days=args.days)
    ctrl_result = simulate_symmetry_control()

    # 2. SRP
    header('2. RADIATION PRESSURE')
    forces = srp_all_vertices(r=r, A_panel=args.area)
    net_f  = srp_net_force(r=r, A_panel=args.area)
    drift  = drift_vs_time(r=r, A_i=args.area, m_i=args.mass)
    lum    = luminosity_perturbation(r=r, gamma=0.001)  # CORRECTED gamma
    row('SRP per S/C [N]', f'{np.linalg.norm(forces[0]):.4f}')
    row('Net force (swarm) [N]', f'{np.linalg.norm(net_f):.2e}')
    row('Drift 10 yr [km]', f'{drift["drift_km"][-1]:.3f}')

    # 3. Thermal (CORRECTED)
    header('3. THERMAL MANAGEMENT [CORRECTED — ×3.33 fix]')
    th_sum = swarm_thermal_summary(r=r, A_panel=args.area, eta_pv=args.eta_pv)
    r_range = np.linspace(0.5*AU, 2.0*AU, 150)
    th_prof = thermal_profile_vs_radius(r_range, A_panel=args.area, eta_pv=args.eta_pv)
    eta_rng = np.linspace(0.05, 0.60, 100)
    eta_sns = eta_pv_sensitivity(eta_rng, r=r, A_panel=args.area)

    row('P_intercept per S/C [MW]', f'{th_sum["P_intercept_W"]/1e6:.3f}')
    row('P_harvest per S/C [MW]',   f'{th_sum["P_harvest_each_W"]/1e6:.3f}')
    row('P_waste per S/C [MW] ✓',   f'{th_sum["P_waste_each_W"]/1e6:.3f}  (was 2.86 MW in v1.0)')
    row('A_rad per S/C [m²] ✓',     f'{th_sum["A_rad_each_m2"]:.1f}  (was 2188 m² in v1.0)')
    row('Radiator/Panel ratio',      f'{th_sum["rad_to_panel_ratio"]:.2f}  (engineering constraint!)')
    row('T_equilibrium [K]',         f'{th_sum["T_equilibrium_K"]:.1f}')
    row('Total harvest [GW]',        f'{th_sum["P_harvest_total_W"]/1e9:.4f}')

    # 4. Beam
    header('4. BEAM RELAY [CORRECTED — actual vertex paths]')
    eff_curve   = efficiency_vs_distance()
    geo_comp    = geometry_comparison(r=r)
    multi_hop   = multi_hop_vs_single(r=r)
    relay_chain = relay_chain_simulation(r=r)

    for g,e in geo_comp.items():
        row(g, f'{e*100:.1f} %')
    row('Adjacent 1-hop η',      f'{multi_hop["eta_direct_adjacent"]*100:.1f}%')
    row('Antipodal 2-hop η ✓',   f'{multi_hop["eta_2hop_antipodal"]*100:.1f}%  (corrected routing)')
    row('vs Direct antipodal',   f'{multi_hop["eta_direct_antipodal"]*100:.1f}%  ({multi_hop["improvement_factor"]:.2f}× better)')

    # 5. Figures
    if not args.no_plots:
        header('5. GENERATING CORRECTED FIGURES')
        from dyson_octahedron.visualization import (
            plot_3d_orbit, plot_lyapunov, plot_radiation_pressure,
            plot_thermal, plot_beam_efficiency, plot_symmetry_control,
            plot_summary_dashboard)

        tasks = [
            ('fig1_3d_orbit.png',  lambda: plot_3d_orbit(r=args.radius, save_path=f'{outdir}/fig1_3d_orbit.png')),
            ('fig2_lyapunov.png',  lambda: plot_lyapunov(sim_result, (radii,freq1,freq2), save_path=f'{outdir}/fig2_lyapunov.png')),
            ('fig3_radiation.png', lambda: plot_radiation_pressure(drift, lum, forces, save_path=f'{outdir}/fig3_radiation.png')),
            ('fig4_thermal.png',   lambda: plot_thermal(th_prof, eta_sns, save_path=f'{outdir}/fig4_thermal.png')),
            ('fig5_beam.png',      lambda: plot_beam_efficiency(eff_curve, geo_comp, multi_hop, relay_chain, save_path=f'{outdir}/fig5_beam.png')),
            ('fig6_symmetry.png',  lambda: plot_symmetry_control(ctrl_result, save_path=f'{outdir}/fig6_symmetry.png')),
            ('fig7_dashboard.png', lambda: plot_summary_dashboard(th_sum, geo_comp, 0.0, net_f, save_path=f'{outdir}/fig7_dashboard.png')),
        ]
        for fname, fn in tasks:
            fn()
            print(f'    ✓ {fname}')

    header('DONE')
    print(f'  Elapsed: {time.time()-t0:.1f}s | Figures → {outdir}/')

if __name__ == '__main__':
    main()
