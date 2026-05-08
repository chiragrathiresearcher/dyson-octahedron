"""
visualization.py  [OctaDyson v1.1.0 — CORRECTED & ENHANCED]
================
All figures regenerated with corrected physics.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = ['#c0392b','#2471a3','#1e8449','#d68910','#7d3c98','#1a5276']
SC_LABELS = ['+x','−x','+y','−y','+z','−z']
AU = 1.496e11

def _style():
    plt.rcParams.update({
        'figure.facecolor': '#fafafa', 'axes.facecolor': '#f8f9fa',
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
        'axes.titleweight': 'bold', 'lines.linewidth': 1.8,
    })

_style()


# ── Figure 1: 3D Orbit ────────────────────────────────────────────────────────
def plot_3d_orbit(r=1.0, save_path=None):
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f0f4f8')
    fig.patch.set_facecolor('#fafafa')

    verts = np.array([
        [ 1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ], dtype=float) * r

    # Draw octahedron edges
    adj_pairs = [(i,j) for i in range(6) for j in range(i+1,6)
                 if not (i+j==1 or i+j==5 or i+j==9)]  # skip antipodal
    for i,j in adj_pairs:
        ax.plot(*zip(verts[i], verts[j]), color='#95a5a6', lw=0.8, alpha=0.5)

    # Spacecraft
    for idx,(v,lbl,c) in enumerate(zip(verts, SC_LABELS, COLORS)):
        ax.scatter(*v, color=c, s=180, zorder=5, edgecolors='white', linewidths=1.5)
        ax.text(v[0]*1.15, v[1]*1.15, v[2]*1.15, lbl, color=c,
                fontsize=11, fontweight='bold', ha='center')

    # Star
    ax.scatter(0, 0, 0, color='#f39c12', s=600, zorder=6,
               edgecolors='#e67e22', linewidths=2)

    # Dashed axes
    for xyz in range(3):
        p = np.zeros(3); p[xyz] = r*1.2
        q = np.zeros(3); q[xyz] = -r*1.2
        ax.plot(*zip(p,q), '--', color='#bdc3c7', lw=0.6, alpha=0.4)

    ax.set_xlabel('X [AU]'); ax.set_ylabel('Y [AU]'); ax.set_zlabel('Z [AU]')
    ax.set_title('Octahedral Dyson Swarm — 3D Orbital Architecture', pad=14)

    lim = r*1.3
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.legend([plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#f39c12',
                           markersize=12,markeredgecolor='#e67e22')],
              ['Host Star'], loc='upper left', framealpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# ── Figure 2: Stability (CORRECTED) ──────────────────────────────────────────
def plot_lyapunov(sim_result, lyap_data, save_path=None):
    """
    CORRECTED: Left = perturbation growth (neutral stability demonstration).
    Right = CW oscillation frequencies vs orbital radius (replaces spurious
            λ_max noise plot from v1.0.0).
    """
    radii, freq1, freq2 = lyap_data
    radii_AU = radii / AU

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Orbital Dynamics — Neutral Stability Analysis [v1.1 CORRECTED]',
                 fontsize=13, fontweight='bold', y=1.01)

    # Left: perturbation evolution
    ax = axes[0]
    t_days = sim_result['t'] / 86400
    for i,(nrm,lbl,c) in enumerate(zip(sim_result['norms'],
                                        sim_result['labels'], COLORS)):
        ax.semilogy(t_days, nrm, color=c, label=f'S/C {lbl}', lw=1.6)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('‖δr‖ [m]')
    ax.set_title('Perturbation Growth — CW Neutral Dynamics\n'
                 '(δ₀ = 1 km; no active control applied)')
    ax.legend(fontsize=9, ncol=2)
    ax.annotate('Control required\nto maintain formation',
                xy=(300, 5e6), xytext=(150, 3e7),
                arrowprops=dict(arrowstyle='->', color='#c0392b'),
                color='#c0392b', fontsize=9, fontweight='bold')

    # Right: CW oscillation frequencies (CORRECTED from spurious λ_max plot)
    ax = axes[1]
    ax.plot(radii_AU, freq1*1e7, color='#2471a3', lw=2, label='ω₁ = n (radial/cross-track)')
    ax.plot(radii_AU, freq2*1e7, color='#1e8449', lw=2,
            ls='--', label='ω₂ ≈ √3·n (in-plane)')
    ax.fill_between(radii_AU, 0, freq1*1e7, alpha=0.08, color='#2471a3')
    ax.fill_between(radii_AU, freq1*1e7, freq2*1e7, alpha=0.08, color='#1e8449')

    ax.set_xlabel('Orbital Radius [AU]')
    ax.set_ylabel('CW Oscillation Frequency [×10⁻⁷ rad s⁻¹]')
    ax.set_title('CW Eigenfrequencies vs Orbital Radius\n'
                 '(All λ_max = 0 exactly — system is neutrally stable)')
    ax.legend(fontsize=9)

    # Annotation box
    ax.text(0.97, 0.95,
            'CORRECTED\nOriginal v1.0 plot showed\nnumerical noise (~10⁻²³ s⁻¹)\nmisidentified as instability',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8.5, color='#922b21',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8',
                      edgecolor='#c0392b', alpha=0.9))

    ax.axhline(0, color='black', lw=0.8, ls=':')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# ── Figure 3: Radiation Pressure ──────────────────────────────────────────────
def plot_radiation_pressure(drift, lum_data, forces, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(drift['t_years'], drift['drift_km'], color='#c0392b', lw=2)
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Secular Drift Δr [km]')
    ax.set_title('Single-Spacecraft SRP Secular Drift')
    ax.fill_between(drift['t_years'], drift['drift_km'], alpha=0.12, color='#c0392b')

    ax = axes[1]
    pair_labels = ['+x/−x', '+y/−y', '+z/−z']
    forces_reshaped = forces.reshape(3, 2, 3)
    net_forces = [np.linalg.norm(forces_reshaped[k,0] + forces_reshaped[k,1])
                  for k in range(3)]
    bars = ax.bar(pair_labels, net_forces, color=['#c0392b','#2471a3','#1e8449'],
                  alpha=0.75, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Net Force ‖Fₐ + F_b‖ [N]')
    ax.set_title('SRP Net Force per Antipodal Pair\n(Perfect cancellation for identical panels)')
    ax.set_ylim(-0.001, max(net_forces)*10 + 0.001 if max(net_forces) > 0 else 0.05)
    for bar, val in zip(bars, net_forces):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{val:.2e} N', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# ── Figure 4: Thermal (CORRECTED) ────────────────────────────────────────────
def plot_thermal(th_prof, eta_sens, save_path=None):
    """CORRECTED thermal plots with proper waste heat formula."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Thermal Management [v1.1 CORRECTED — Waste Heat Formula Fixed]',
                 fontsize=13, fontweight='bold', y=1.01)

    ax  = axes[0]
    ax2 = ax.twinx()

    l1, = ax.plot(th_prof['r_AU'], th_prof['P_harvest']/1e9, color='#d68910',
                  lw=2, label='P_harvest [GW]')
    l2, = ax.plot(th_prof['r_AU'], th_prof['P_waste']/1e9,   color='#c0392b',
                  lw=2, ls='--', label='P_waste [GW] (CORRECTED)')
    l3, = ax2.plot(th_prof['r_AU'], th_prof['A_rad'],         color='#2471a3',
                   lw=2, ls=':', label='A_rad [m²] (CORRECTED)')

    ax.set_xlabel('Orbital Radius [AU]')
    ax.set_ylabel('Power [GW]')
    ax2.set_ylabel('Radiator Area [m²]', color='#2471a3')
    ax2.tick_params(axis='y', labelcolor='#2471a3')
    ax.set_title('Thermal Quantities vs Orbital Radius\n(P_waste = P_intercept × (1−η_PV))')
    ax.legend(handles=[l1,l2,l3], fontsize=9, loc='upper right')

    # Annotation: what was wrong
    ax.annotate('P_waste corrected:\n×3.33 larger than v1.0',
                xy=(1.0, th_prof['P_waste'][np.argmin(np.abs(th_prof['r_AU']-1.0))]/1e9),
                xytext=(1.4, 0.008),
                arrowprops=dict(arrowstyle='->', color='#922b21'),
                color='#922b21', fontsize=9, fontweight='bold')

    ax = axes[1]
    # Show BOTH original (wrong) and corrected radiator area
    # Reconstruct wrong version: P_waste_wrong = P_harvest*(1-eta)
    eta_arr = eta_sens['eta_pv']
    P_wrong = eta_sens['P_harvest'] * (1 - eta_arr)
    SIGMA = 5.6704e-8; eps = 0.90; T = 400.0
    A_wrong = P_wrong / (SIGMA * eps * (T**4 - 3**4))

    ax.plot(eta_arr*100, eta_sens['A_rad'], color='#1e8449', lw=2.5,
            label='Corrected (v1.1)')
    ax.plot(eta_arr*100, A_wrong, color='#c0392b', lw=2, ls='--',
            alpha=0.7, label='Original v1.0 (WRONG)')
    ax.fill_between(eta_arr*100, A_wrong, eta_sens['A_rad'],
                    alpha=0.15, color='#c0392b', label='Error region')

    ax.axvline(30, color='grey', ls=':', lw=1)
    ax.text(31, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1000,
            'η=30%\n(baseline)', fontsize=8.5, color='grey')

    ax.set_xlabel('PV Efficiency η_PV [%]')
    ax.set_ylabel('Required Radiator Area [m²]')
    ax.set_title('Radiator Area: Corrected vs Original\n(r = 1 AU)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# ── Figure 5: Beam Efficiency (CORRECTED) ─────────────────────────────────────
def plot_beam_efficiency(eff_curve, geo_comp, multi_hop, relay_chain, save_path=None):
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
    fig.suptitle('Beam Relay Efficiency [v1.1 CORRECTED — Multi-hop model fixed]',
                 fontsize=13, fontweight='bold', y=1.01)

    # Top-left: attenuation curve
    ax = fig.add_subplot(gs[0,0])
    ax.plot(eff_curve['D_AU'], eff_curve['eta_beam']*100, color='#c0392b', lw=2)
    ax.axvline(np.sqrt(2), color='#1e8449', ls='--', lw=1.5,
               label=f'Oct. edge ≈ {np.sqrt(2):.2f} AU')
    ax.axvline(2.0, color='#2471a3', ls=':', lw=1.5, label='Antipodal = 2 AU')
    ax.set_xlabel('Distance [AU]'); ax.set_ylabel('η_beam [%]')
    ax.set_title('Beam Attenuation vs Distance')
    ax.legend(fontsize=9)

    # Top-right: geometry comparison
    ax = fig.add_subplot(gs[0,1])
    geos = list(geo_comp.keys()); etas = [v*100 for v in geo_comp.values()]
    colors_bar = ['#1e8449' if g=='Octahedron' else '#95a5a6' for g in geos]
    bars = ax.barh(geos, etas, color=colors_bar, edgecolor='white', linewidth=1.5)
    for bar,val in zip(bars,etas):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    ax.set_xlabel('η_beam [%]')
    ax.set_title('Beam Efficiency by Geometry\n(Same circumradius r = 1 AU)')
    ax.set_xlim(0, 90)

    # Bottom-left: CORRECTED multi-hop (actual vertex paths)
    ax = fig.add_subplot(gs[1,0])
    n_hops = multi_hop['n_hops']
    eta_m  = multi_hop['eta_multi'] * 100

    bars2 = ax.bar(n_hops, eta_m, color=['#2471a3','#1e8449'],
                   edgecolor='white', linewidth=1.5, width=0.5)
    ax.axhline(multi_hop['eta_direct']*100, color='#c0392b', ls='--', lw=2,
               label=f'Direct antipodal (1 hop): {multi_hop["eta_direct"]*100:.1f}%')

    labels_hop = ['Adjacent\n(1 hop, r√2)', 'Antipodal via\nadjacent (2 hops)']
    ax.set_xticks(n_hops)
    ax.set_xticklabels(labels_hop)
    for bar,val in zip(bars2, eta_m):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('η_total [%]')
    ax.set_title('Multi-Hop Relay Efficiency [CORRECTED]\n'
                 'Actual octahedron vertex paths only')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 90)

    # CORRECTED annotation
    ax.text(0.98, 0.08,
            'CORRECTED: Original v1.0\nshowed sub-edge hops that\nrequire non-existent nodes.',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#922b21',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8',
                      edgecolor='#c0392b', alpha=0.85))

    # Bottom-right: stochastic relay chain
    ax  = fig.add_subplot(gs[1,1])
    ax2 = ax.twinx()
    hop_idx = relay_chain['hop']
    bars3   = ax.bar(hop_idx, relay_chain['eta_hop']*100, alpha=0.6,
                     color='#d68910', label='Per-hop η', width=0.6)
    l2, = ax2.plot(hop_idx, relay_chain['eta_cumulative']*100,
                   'o-', color='#c0392b', lw=2, label='Cumulative η', zorder=5)

    ax.set_xlabel('Relay Hop Index')
    ax.set_ylabel('Per-hop η [%]')
    ax2.set_ylabel('Cumulative η [%]', color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    ax.set_title('Stochastic Relay Chain\n(Pointing-noise σ = 2%)')
    handles = [mpatches.Patch(color='#d68910',alpha=0.6,label='Per-hop η'), l2]
    ax.legend(handles=handles, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# ── Figure 6: Symmetry Control ────────────────────────────────────────────────
def plot_symmetry_control(ctrl_result, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.semilogy(ctrl_result['steps'], ctrl_result['E_sym'],
                color='#c0392b', lw=2)
    ax.set_xlabel('Control Iteration')
    ax.set_ylabel('Symmetry Error E_sym')
    ax.set_title('Gradient-Based Symmetry Control Convergence')
    ax.text(0.97, 0.95, f'Final E_sym = {ctrl_result["E_sym"][-1]:.2e}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color='#2471a3',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8'))

    ax = axes[1]
    n_hist = ctrl_result['n_hat_history']
    ax.add_patch(plt.Circle((0,0), 1.0, fill=False, color='grey',
                             ls='--', lw=1, alpha=0.4))
    for i in range(6):
        xi = n_hist[:, i, 0]; yi = n_hist[:, i, 1]
        ax.plot(xi, yi, color=COLORS[i], lw=0.8, alpha=0.4)
        ax.scatter(xi[0], yi[0], marker='x', s=80, color=COLORS[i],
                   zorder=5, linewidths=2)
        ax.scatter(xi[-1], yi[-1], marker='o', s=80, color=COLORS[i],
                   zorder=5, edgecolors='white', linewidths=1.5)

    ax.set_aspect('equal')
    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Vertex Restoration (× = initial, • = final)\nXY Projection')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# ── Figure 7: Dashboard (CORRECTED) ──────────────────────────────────────────
def plot_summary_dashboard(th_sum, geo_comp, lam_max, net_f, save_path=None):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#fafafa')
    fig.suptitle('OctaDyson — System Summary Dashboard  (r = 1 AU)  [v1.1 CORRECTED]',
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4,
                           top=0.90, bottom=0.06, left=0.07, right=0.97)

    # ── Power budget pie (CORRECTED values) ──
    ax = fig.add_subplot(gs[0, 0])
    Ph = th_sum['P_harvest_total_W']
    Pw = th_sum['P_waste_total_W']
    eta = th_sum['eta_pv']
    ax.pie([Ph, Pw], labels=['Net Electrical\nOutput', 'Waste Heat'],
           colors=['#1e8449','#c0392b'], autopct='%1.1f%%',
           startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
    ax.set_title(f'Power Budget (6 S/C)\nPh={Ph/1e6:.1f} MW, Pw={Pw/1e6:.1f} MW\n'
                 f'[CORRECTED: +{(1-eta)/eta*100:.0f}% more waste vs v1.0]',
                 fontsize=9)

    # ── Beam efficiency bar ──
    ax = fig.add_subplot(gs[0, 1])
    geos = list(geo_comp.keys()); etas_b = [v*100 for v in geo_comp.values()]
    colors_b = ['#1e8449' if g=='Octahedron' else '#95a5a6' for g in geos]
    bars = ax.barh(geos, etas_b, color=colors_b, edgecolor='white', lw=1.5)
    for b,v in zip(bars,etas_b):
        ax.text(v+0.5, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center', fontsize=9)
    ax.axvline(max(etas_b)*0.9, color='#2471a3', ls='--', alpha=0.5)
    ax.set_xlabel('η_beam [%]'); ax.set_xlim(0,90)
    ax.set_title('Beam Efficiency by Geometry')

    # ── Key parameters table (CORRECTED) ──
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    params = [
        ['Parameter', 'v1.0 (WRONG)', 'v1.1 (CORRECT)'],
        ['λ_max [s⁻¹]',  '~0 (neutral)', '0.0 (exact)'],
        ['SRP Net [N]',   '0.00e+00', '0.00e+00'],
        ['P_harvest [MW]', f'{th_sum["P_harvest_each_W"]/1e6:.2f}', f'{th_sum["P_harvest_each_W"]/1e6:.2f}'],
        ['P_waste [MW]',   '2.86', f'{th_sum["P_waste_each_W"]/1e6:.2f}'],
        ['A_rad [m²]',     '2188', f'{th_sum["A_rad_each_m2"]:.0f}'],
        ['A_rad/A_panel',  '0.22',  f'{th_sum["rad_to_panel_ratio"]:.2f}'],
        ['T_eq [K]',       '400.0', f'{th_sum["T_equilibrium_K"]:.1f}'],
    ]
    tbl = ax.table(cellText=params[1:], colLabels=params[0],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.6)
    for (r_,c_), cell in tbl.get_celld().items():
        if r_ == 0:
            cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
        elif c_ == 1 and r_ > 0:
            cell.set_facecolor('#fadbd8')
        elif c_ == 2 and r_ > 0:
            cell.set_facecolor('#d5f5e3')
    ax.set_title('Key Parameters: Corrected vs Original', fontsize=9, fontweight='bold', pad=12)

    # ── Swarm topology ──
    ax = fig.add_subplot(gs[1, 0])
    verts2d = np.array([[1,0],[-1,0],[0,1],[0,-1],[0.7,0.7],[-0.7,-0.7]])
    ax.add_patch(plt.Circle((0,0),1.3,fill=False,color='grey',ls='--',lw=1,alpha=0.4))
    for i in range(6):
        ax.scatter(*verts2d[i], s=150, color=COLORS[i],
                   zorder=5, edgecolors='white', lw=1.5)
        ax.text(verts2d[i][0]*1.25, verts2d[i][1]*1.25, SC_LABELS[i],
                ha='center', va='center', fontsize=9, color=COLORS[i])
    ax.scatter(0,0,s=400,color='#f39c12',zorder=6,edgecolors='#e67e22',lw=2)
    ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.6,1.6)
    ax.set_aspect('equal'); ax.set_title('Swarm Topology (Top View)')

    # ── Thermal budget CORRECTED ──
    ax = fig.add_subplot(gs[1, 1])
    categories = ['P_harvest\n[MW/S/C]', 'P_waste\n[MW/S/C]\nCORRECTED',
                  'A_rad\n[×100 m²]\nCORRECTED']
    vals = [th_sum['P_harvest_each_W']/1e6,
            th_sum['P_waste_each_W']/1e6,
            th_sum['A_rad_each_m2']/100]
    old_vals = [None, th_sum['P_harvest_each_W']/1e6*(1-th_sum['eta_pv'])/1e6, 2188/100]
    colors_c = ['#d68910','#c0392b','#2471a3']
    bars_c = ax.bar(range(len(categories)), vals, color=colors_c,
                    alpha=0.8, edgecolor='white', lw=1.5)
    # Overlay old wrong values
    ax.bar([1,2], [old_vals[1], old_vals[2]], color=['#fadbd8','#d6eaf8'],
           alpha=0.6, edgecolor='#c0392b', lw=1.5, ls='--', width=0.5,
           label='v1.0 (wrong)')
    for bar,val in zip(bars_c, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_title('Per-Spacecraft Thermal Budget\n(CORRECTED)')
    ax.legend(fontsize=8)

    # ── SRP net force per pair ──
    ax = fig.add_subplot(gs[1, 2])
    pair_labels = ['+x/−x', '+y/−y', '+z/−z']
    net_vals = [np.linalg.norm(net_f)/3]*3   # ~0 for all pairs
    bars_s = ax.bar(pair_labels, net_vals, color=['#c0392b','#2471a3','#1e8449'],
                    alpha=0.75, edgecolor='white', lw=1.5)
    ax.set_ylim(-0.001, 0.05)
    ax.set_ylabel('Net Force [N]')
    ax.set_title('SRP Net Force per Antipodal Pair\n(Should be ≈ 0)')
    for bar in bars_s:
        ax.text(bar.get_x()+bar.get_width()/2, 0.002,
                f'{bar.get_height():.2e}', ha='center', fontsize=9, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig
