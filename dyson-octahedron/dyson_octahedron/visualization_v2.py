"""
visualization_v2.py — OctaDyson v2.0
======================================
Ten publication-quality figures covering all physics modules.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D

AU = 1.496e11

# ── Palette ───────────────────────────────────────────────────────────────────
C  = ['#c0392b','#2471a3','#1e8449','#d68910','#7d3c98','#1a5276']
BG = '#f8f9fa'; AX = '#ffffff'; GR = '#e8eaed'

def _base():
    plt.rcParams.update({
        'figure.facecolor' : BG,  'axes.facecolor'  : AX,
        'axes.grid'        : True,'grid.alpha'      : 0.25,
        'grid.linewidth'   : 0.6, 'axes.spines.top' : False,
        'axes.spines.right': False,'font.family'    : 'DejaVu Sans',
        'font.size'        : 10,  'axes.labelsize'  : 11,
        'axes.titlesize'   : 12,  'axes.titleweight': 'bold',
        'lines.linewidth'  : 2.0, 'axes.edgecolor'  : '#cccccc',
    })
_base()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — 3D Architecture (enhanced)
# ══════════════════════════════════════════════════════════════════════════════
def fig_architecture(save=None):
    fig = plt.figure(figsize=(10,9), facecolor=BG)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#eef2f7')

    V = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
    L = ['+x','−x','+y','−y','+z','−z']

    # Draw 12 edges
    for i in range(6):
        for j in range(i+1,6):
            if abs(i-j)!=1 or (i in [1,3,5]):  # skip antipodal
                if i+j not in [1,5,9]:
                    ax.plot(*zip(V[i],V[j]),color='#aab4c8',lw=0.9,alpha=0.6)

    # Antipodal dashed
    for p,q in [(0,1),(2,3),(4,5)]:
        ax.plot(*zip(V[p],V[q]),'--',color='#c0392b',lw=0.7,alpha=0.4)

    # Orbital sphere wireframe
    u,v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    xs,ys,zs = np.cos(u)*np.sin(v),np.sin(u)*np.sin(v),np.cos(v)
    ax.plot_surface(xs,ys,zs,alpha=0.04,color='#2471a3',linewidth=0)

    # Spacecraft
    for vi,lbl,c in zip(V,L,C):
        ax.scatter(*vi,color=c,s=260,zorder=8,edgecolors='white',linewidths=2)
        off = vi*1.18
        ax.text(*off,lbl,color=c,fontsize=12,fontweight='bold',ha='center',va='center',
                zorder=9)
        # Orbit arcs suggestion
        ax.plot([0,vi[0]],[0,vi[1]],[0,vi[2]],'--',color=c,lw=0.6,alpha=0.3)

    # Host star
    ax.scatter(0,0,0,color='#f39c12',s=800,zorder=10,
               edgecolors='#e67e22',linewidths=2.5)
    ax.text(0,0,-0.18,'☆ Host Star',ha='center',fontsize=10,
            color='#e67e22',fontweight='bold',zorder=11)

    ax.set_xlabel('X [AU]',labelpad=8); ax.set_ylabel('Y [AU]',labelpad=8)
    ax.set_zlabel('Z [AU]',labelpad=8)
    ax.set_title('Octahedral Dyson Swarm — 3D Orbital Architecture\n'
                 'Six spacecraft at ±x, ±y, ±z vertices of a regular octahedron',pad=14)
    ax.set_xlim(-1.4,1.4); ax.set_ylim(-1.4,1.4); ax.set_zlim(-1.4,1.4)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#f39c12',
                      markersize=12,markeredgecolor='#e67e22',label='Host Star'),
               Line2D([0],[0],ls='--',color='#c0392b',lw=1.2,label='Antipodal axis')]
    ax.legend(handles=handles,loc='upper left',framealpha=0.85,fontsize=9)
    plt.tight_layout()
    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Orbital Dynamics (Perturbation + Eigenfrequency)
# ══════════════════════════════════════════════════════════════════════════════
def fig_orbital(sim, lyap_data, save=None):
    radii, freq1, freq2 = lyap_data
    fig, axes = plt.subplots(1,2,figsize=(14,5.5),facecolor=BG)
    fig.suptitle('Orbital Dynamics — Hill–Clohessy–Wiltshire Analysis',
                 fontsize=13,fontweight='bold',y=1.01)

    ax = axes[0]
    t  = sim['t']/86400
    for nrm,lbl,c in zip(sim['norms'],sim['labels'],C):
        ax.semilogy(t,nrm,color=c,label=f'S/C {lbl}',lw=1.8)
    ax.set_xlabel('Time [days]'); ax.set_ylabel('‖δr‖ [m]')
    ax.set_title('Perturbation Growth Under Neutral CW Dynamics\n(δ₀ = 1 km; no active control)')
    ax.legend(fontsize=9,ncol=2)
    ax.fill_between(t,sim['norms'].min(0),sim['norms'].max(0),alpha=0.07,color='grey')
    ax.axhline(1e3,color='grey',ls=':',lw=1)
    ax.text(350,1.3e3,'δ₀ = 1 km',fontsize=8,color='grey',ha='right')

    ax = axes[1]
    r_AU = radii/AU
    ax.plot(r_AU,freq1*1e7,color='#2471a3',lw=2.2,label='ω₁ = n  (radial)')
    ax.plot(r_AU,freq2*1e7,color='#1e8449',lw=2.2,ls='--',label='ω₂ ≈ √3·n  (in-plane)')
    ax.fill_between(r_AU,0,freq1*1e7,alpha=0.1,color='#2471a3')
    ax.fill_between(r_AU,freq1*1e7,freq2*1e7,alpha=0.1,color='#1e8449')
    ax.axvline(1.0,color='grey',ls=':',lw=1); ax.text(1.03,ax.get_ylim()[1]*0.9,'1 AU',fontsize=8,color='grey')
    ax.set_xlabel('Orbital Radius [AU]')
    ax.set_ylabel('CW Frequency [×10⁻⁷ rad s⁻¹]')
    ax.set_title('CW Eigenfrequency Spectrum vs Orbital Radius\n(λ_max = 0 exactly — neutrally stable)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Station-Keeping & Delta-V Budget (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def fig_station_keeping(sk_data, fuel_summary, isp_sweep, save=None):
    fig = plt.figure(figsize=(14,10),facecolor=BG)
    gs  = gridspec.GridSpec(2,2,hspace=0.4,wspace=0.35,
                            top=0.91,bottom=0.07,left=0.08,right=0.97)
    fig.suptitle('Station-Keeping — Delta-V Budget & Propellant Analysis',
                 fontsize=13,fontweight='bold')

    # Top-left: delta-v vs radius
    ax = fig.add_subplot(gs[0,0])
    ax.plot(sk_data['r_AU'],sk_data['dv_mps_yr'],color='#c0392b',lw=2.5)
    ax.fill_between(sk_data['r_AU'],0,sk_data['dv_mps_yr'],alpha=0.12,color='#c0392b')
    ax.axvline(1.0,color='grey',ls=':',lw=1); ax.text(1.03,sk_data['dv_mps_yr'].max()*0.85,'1 AU',fontsize=8.5,color='grey')
    ax.set_xlabel('Orbital Radius [AU]'); ax.set_ylabel('Annual ΔV [m s⁻¹ yr⁻¹]')
    ax.set_title('Annual Station-Keeping ΔV vs Orbital Radius')
    # Mark 1 AU value
    dv1 = np.interp(1.0, sk_data['r_AU'], sk_data['dv_mps_yr'])
    ax.scatter([1.0],[dv1],color='#c0392b',s=80,zorder=5)
    ax.annotate(f'{dv1:.1f} m/s/yr\n@ 1 AU',xy=(1.0,dv1),xytext=(1.3,dv1*1.2),
                arrowprops=dict(arrowstyle='->',color='#c0392b'),fontsize=9,color='#c0392b')

    # Top-right: propellant mass fraction vs Isp
    ax = fig.add_subplot(gs[0,1])
    ax.plot(isp_sweep['Isp'],isp_sweep['prop_frac']*100,color='#2471a3',lw=2.5)
    ax.fill_between(isp_sweep['Isp'],0,isp_sweep['prop_frac']*100,alpha=0.12,color='#2471a3')
    ax.axvline(3000,color='#7d3c98',ls='--',lw=1.5,label='Ion thruster (3000 s)')
    ax.axvline(450, color='#d68910',ls='--',lw=1.5,label='Chemical (450 s)')
    pf_3k = np.interp(3000, isp_sweep['Isp'], isp_sweep['prop_frac']*100)
    pf_ch = np.interp(450,  isp_sweep['Isp'], isp_sweep['prop_frac']*100)
    ax.scatter([3000,450],[pf_3k,pf_ch],color=['#7d3c98','#d68910'],s=80,zorder=5)
    ax.set_xlabel('Specific Impulse Isp [s]')
    ax.set_ylabel('10-year Propellant Fraction [%]')
    ax.set_title('Propellant Mass Fraction vs Thruster Performance\n(10-year mission, r=1 AU)')
    ax.legend(fontsize=9)

    # Bottom-left: mass breakdown waterfall
    ax = fig.add_subplot(gs[1,0])
    labels = ['PV Panels','Radiators','Structure','Avionics','Propellant']
    values = [fuel_summary['m_pv_kg'],fuel_summary['m_radiator_kg'],
              fuel_summary['m_structure_kg'],fuel_summary['m_avionics_kg'],
              fuel_summary['m_propellant_kg']]
    colors_b = ['#d68910','#c0392b','#95a5a6','#2471a3','#7d3c98']
    bars = ax.bar(labels,values,color=colors_b,edgecolor='white',linewidth=1.5,alpha=0.85)
    for bar,val in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f'{val:.0f} kg',ha='center',fontsize=9,fontweight='bold')
    ax.set_ylabel('Mass [kg]')
    ax.set_title(f'Spacecraft Mass Budget\n(Total wet mass: {fuel_summary["m_wet_kg"]:.0f} kg)')
    ax.tick_params(axis='x',rotation=15)

    # Bottom-right: correction frequency / control timeline
    ax = fig.add_subplot(gs[1,1])
    radii_AU = sk_data['r_AU']
    tol_vals = [100, 1000, 10000]   # metres
    for tol in tol_vals:
        from dyson_octahedron.station_keeping import lqr_control_cost
        t_corr = np.array([lqr_control_cost(r*AU, 1e4, 1e3, tol)['T_corr_days']
                           for r in radii_AU])
        ax.semilogy(radii_AU, t_corr, lw=2, label=f'tol = {tol/1e3:.0f} km')
    ax.set_xlabel('Orbital Radius [AU]')
    ax.set_ylabel('Correction Interval [days]')
    ax.set_title('LQR Control: Correction Interval vs Tolerance\n(per spacecraft)')
    ax.legend(fontsize=9)

    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Thermal Management
# ══════════════════════════════════════════════════════════════════════════════
def fig_thermal(th_prof, eta_sens, save=None):
    fig, axes = plt.subplots(1,2,figsize=(14,5.5),facecolor=BG)
    fig.suptitle('Thermal Management — Waste Heat Rejection Analysis',fontsize=13,fontweight='bold',y=1.01)

    ax = axes[0]; ax2 = ax.twinx()
    l1,=ax.plot(th_prof['r_AU'],th_prof['P_harvest']/1e9,color='#d68910',lw=2.2,label='P_harvest [GW]')
    l2,=ax.plot(th_prof['r_AU'],th_prof['P_waste']/1e9,  color='#c0392b',lw=2.2,ls='--',label='P_waste [GW]')
    l3,=ax2.plot(th_prof['r_AU'],th_prof['A_rad'],        color='#2471a3',lw=2.2,ls=':',label='A_rad [m²]')
    ax.set_xlabel('Orbital Radius [AU]'); ax.set_ylabel('Power [GW]')
    ax2.set_ylabel('Radiator Area [m²]',color='#2471a3')
    ax2.tick_params(axis='y',labelcolor='#2471a3')
    ax.set_title('Thermal Quantities vs Orbital Radius\nP_waste = P_intercept × (1−η_PV)')
    ax.legend(handles=[l1,l2,l3],fontsize=9,loc='upper right')

    ax = axes[1]
    ax.plot(eta_sens['eta_pv']*100, eta_sens['A_rad'],   color='#1e8449',lw=2.5,label='A_rad required')
    ax.plot(eta_sens['eta_pv']*100, 1e4*np.ones_like(eta_sens['eta_pv']),
            '--',color='#c0392b',lw=1.5,label='Panel area = 10,000 m²')
    # Mark equal-area crossover
    cross_eta = eta_sens['eta_pv'][np.argmin(np.abs(eta_sens['A_rad']-1e4))]
    ax.axvline(cross_eta*100,color='#7d3c98',ls=':',lw=1.5)
    ax.text(cross_eta*100+1,1.1e4,f'Equal area\nη={cross_eta*100:.0f}%',fontsize=8.5,color='#7d3c98')
    ax.axvline(30,color='grey',ls=':',lw=1); ax.text(31,500,'η=30%',fontsize=8,color='grey')
    ax.fill_between(eta_sens['eta_pv']*100, eta_sens['A_rad'], 1e4,
                    where=eta_sens['A_rad']>1e4, alpha=0.1, color='#c0392b',
                    label='A_rad > A_panel (overconstrained)')
    ax.set_xlabel('PV Efficiency η_PV [%]')
    ax.set_ylabel('Required Radiator Area [m²]')
    ax.set_title('Radiator Area vs PV Efficiency (r=1 AU)\nCrossover marks engineering feasibility threshold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Beam Relay
# ══════════════════════════════════════════════════════════════════════════════
def fig_beam(eff_curve, geo_comp, mhop, relay, save=None):
    fig = plt.figure(figsize=(14,11),facecolor=BG)
    gs  = gridspec.GridSpec(2,2,hspace=0.42,wspace=0.35)
    fig.suptitle('CSBPB Beam Relay Efficiency Analysis',fontsize=13,fontweight='bold',y=1.01)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(eff_curve['D_AU'],eff_curve['eta_beam']*100,color='#c0392b',lw=2.2)
    ax.axvline(np.sqrt(2),color='#1e8449',ls='--',lw=1.8,label=f'Oct. edge ≈1.41 AU ({eff_curve["eta_beam"][np.argmin(np.abs(eff_curve["D_AU"]-np.sqrt(2)))]*100:.1f}%)')
    ax.axvline(2.0,color='#2471a3',ls=':',lw=1.8,label=f'Antipodal = 2 AU ({eff_curve["eta_beam"][np.argmin(np.abs(eff_curve["D_AU"]-2.0))]*100:.1f}%)')
    ax.fill_between(eff_curve['D_AU'],0,eff_curve['eta_beam']*100,alpha=0.08,color='#c0392b')
    ax.set_xlabel('Distance [AU]'); ax.set_ylabel('η_beam [%]')
    ax.set_title('Beam Attenuation vs Distance')
    ax.legend(fontsize=8.5)

    ax = fig.add_subplot(gs[0,1])
    geos = list(geo_comp.keys()); etas = [v*100 for v in geo_comp.values()]
    colors_g = ['#1e8449' if g=='Octahedron' else '#95a5a6' for g in geos]
    bars = ax.barh(geos, etas, color=colors_g, edgecolor='white', lw=1.5)
    for bar,val in zip(bars,etas):
        ax.text(val+0.5,bar.get_y()+bar.get_height()/2,f'{val:.1f}%',va='center',fontsize=10)
    ax.set_xlabel('η_beam [%]'); ax.set_title('Single-Hop Efficiency by Geometry')
    ax.set_xlim(0,90)

    ax = fig.add_subplot(gs[1,0])
    scenarios = ['Adjacent\n(1 hop, r√2)','Antipodal\n2-hop via adj.','Antipodal\ndirect (1 hop)']
    vals = [mhop['eta_direct_adjacent']*100, mhop['eta_2hop_antipodal']*100, mhop['eta_direct_antipodal']*100]
    bars2 = ax.bar(range(3),vals,color=['#2471a3','#1e8449','#c0392b'],edgecolor='white',lw=1.5,width=0.55,alpha=0.85)
    for b,v in zip(bars2,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f'{v:.1f}%',ha='center',fontsize=10.5,fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(scenarios,fontsize=9)
    ax.set_ylabel('η [%]'); ax.set_title('Relay Scenarios — Actual Vertex Paths')
    ax.set_ylim(0,85)

    ax = fig.add_subplot(gs[1,1]); ax2=ax.twinx()
    bars3=ax.bar(relay['hop'],relay['eta_hop']*100,alpha=0.55,color='#d68910',label='Per-hop η',width=0.6)
    l2,=ax2.plot(relay['hop'],relay['eta_cumulative']*100,'o-',color='#c0392b',lw=2.2,ms=8,label='Cumulative η')
    for h,v in zip(relay['hop'],relay['eta_cumulative']*100):
        ax2.text(h,v+1.5,f'{v:.1f}%',ha='center',fontsize=8.5,color='#c0392b',fontweight='bold')
    ax.set_xlabel('Relay Hop Index'); ax.set_ylabel('Per-hop η [%]')
    ax2.set_ylabel('Cumulative η [%]',color='#c0392b')
    ax2.tick_params(axis='y',labelcolor='#c0392b')
    ax.set_title('Stochastic Relay Chain (σ_pointing = 2%)')
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color='#d68910',alpha=0.55,label='Per-hop η'),l2],fontsize=9)

    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Symmetry Control
# ══════════════════════════════════════════════════════════════════════════════
def fig_symmetry(ctrl, save=None):
    fig,axes = plt.subplots(1,2,figsize=(13,5.5),facecolor=BG)
    fig.suptitle('Gradient-Based Symmetry Control',fontsize=13,fontweight='bold',y=1.01)

    ax = axes[0]
    ax.semilogy(ctrl['steps'],ctrl['E_sym'],color='#c0392b',lw=2.2)
    ax.fill_between(ctrl['steps'],ctrl['E_sym'],ctrl['E_sym'][-1],alpha=0.08,color='#c0392b')
    ax.set_xlabel('Control Iteration'); ax.set_ylabel('Symmetry Error E_sym')
    ax.set_title('Convergence of Gradient Descent Controller')
    ax.text(0.97,0.95,f'E_sym₀ = {ctrl["E_sym"][0]:.3f}\nE_sym_∞ = {ctrl["E_sym"][-1]:.2e}',
            transform=ax.transAxes,ha='right',va='top',fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4',facecolor='#d6eaf8',edgecolor='#2471a3'))

    ax = axes[1]
    nh = ctrl['n_hat_history']
    ax.add_patch(plt.Circle((0,0),1.0,fill=False,color='#aab4c8',ls='--',lw=1,alpha=0.6))
    for i in range(6):
        x,y = nh[:,i,0], nh[:,i,1]
        ax.plot(x,y,color=C[i],lw=0.8,alpha=0.5)
        ax.scatter(x[0],y[0],marker='x',s=90,color=C[i],zorder=5,linewidths=2.2)
        ax.scatter(x[-1],y[-1],marker='o',s=90,color=C[i],zorder=5,
                   edgecolors='white',linewidths=1.8)
    ax.set_aspect('equal'); ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Vertex Restoration — XY Projection\n(× initial → ● final)')

    plt.tight_layout()
    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Coverage Map (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def fig_coverage(cov_map, cov_stats, save=None):
    fig = plt.figure(figsize=(14,9),facecolor=BG)
    gs  = gridspec.GridSpec(2,3,hspace=0.4,wspace=0.35,
                            top=0.90,bottom=0.07,left=0.07,right=0.97)
    fig.suptitle('Sky Coverage & Power Delivery Map — Octahedral Swarm (r=1 AU)',
                 fontsize=13,fontweight='bold')

    # Full sky power map
    ax = fig.add_subplot(gs[0,:2])
    LON_d = np.degrees(cov_map['LON'])
    LAT_d = np.degrees(cov_map['LAT'])
    eta   = cov_map['eta_map']*100
    cmap  = plt.cm.YlOrRd
    im    = ax.pcolormesh(LON_d,LAT_d,eta,cmap=cmap,vmin=0,vmax=100,shading='auto')
    plt.colorbar(im,ax=ax,label='Best η_beam to any S/C [%]',shrink=0.85)

    # Mark vertex directions
    V2d = [
        (0,0,'−x'),(180,0,'+x'),(90,0,'+y'),(270,0,'−y'),
        (0,90,'+z'),(0,-90,'−z')
    ]
    for lon,lat,lbl in V2d:
        ax.scatter(lon,lat,s=120,color='white',edgecolors='black',zorder=5,linewidths=1.5)
        ax.text(lon+5,lat+5,lbl,fontsize=8.5,fontweight='bold',color='black',zorder=6)

    ax.set_xlabel('Longitude [°]'); ax.set_ylabel('Latitude [°]')
    ax.set_title(f'Sky Coverage Map — Mean η = {cov_map["mean_eta"]*100:.1f}%,  '
                 f'Coverage >30% = {cov_map["coverage_frac"]*100:.1f}%')
    ax.set_xlim(0,360); ax.set_ylim(-90,90)

    # Redundancy histogram
    ax = fig.add_subplot(gs[0,2])
    redund = cov_stats['redundancy']
    keys   = sorted(redund.keys())
    vals   = [redund[k]*100 for k in keys]
    bars_r = ax.bar([str(k) for k in keys],vals,
                    color=['#c0392b' if k==0 else '#2471a3' if k==1 else '#1e8449' for k in keys],
                    edgecolor='white',lw=1.5,alpha=0.85)
    for b,v in zip(bars_r,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f'{v:.1f}%',ha='center',fontsize=10)
    ax.set_xlabel('Number of visible S/C')
    ax.set_ylabel('Sky fraction [%]')
    ax.set_title('Multi-Source Redundancy\n(η > 10% threshold)')

    # Coverage vs radius
    ax = fig.add_subplot(gs[1,:2])
    cv = cov_stats.get('vs_radius',None)
    if cv is not None:
        ax.plot(cv['r_AU'],cv['mean_eta']*100,color='#2471a3',lw=2.2,label='Mean η [%]')
        ax.plot(cv['r_AU'],cv['frac_above_50pct']*100,color='#1e8449',lw=2.2,ls='--',label='Sky fraction η>50%')
        ax.plot(cv['r_AU'],cv['min_eta']*100,color='#c0392b',lw=2.2,ls=':',label='Min η [%]')
    ax.set_xlabel('Orbital Radius [AU]')
    ax.set_ylabel('[%]')
    ax.set_title('Coverage Quality vs Orbital Radius')
    ax.legend(fontsize=9)

    # Rose chart — which S/C serves which sky fraction
    ax = fig.add_subplot(gs[1,2])
    best_sc = cov_map['best_sc']
    labels  = ['+x','−x','+y','−y','+z','−z']
    counts  = [(best_sc==i).mean()*100 for i in range(6)]
    wedges, texts, autotexts = ax.pie(counts,labels=labels,colors=C,
                                       autopct='%1.1f%%',startangle=45,
                                       wedgeprops={'edgecolor':'white','linewidth':1.5})
    ax.set_title('Sky Fraction Served\nby Each Spacecraft')

    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Mass Budget & Technology Readiness (NEW)
# ══════════════════════════════════════════════════════════════════════════════
def fig_mass_trl(mb, trl, mass_sens, save=None):
    fig = plt.figure(figsize=(14,10),facecolor=BG)
    gs  = gridspec.GridSpec(2,3,hspace=0.45,wspace=0.38,
                            top=0.91,bottom=0.07,left=0.07,right=0.97)
    fig.suptitle('Mass Budget, Specific Power & Technology Readiness',
                 fontsize=13,fontweight='bold')

    # Top-left: mass breakdown donut
    ax = fig.add_subplot(gs[0,0])
    labels = ['PV Panels','Radiators','Structure','Avionics','Propellant']
    vals   = [mb['m_pv_kg'],mb['m_radiator_kg'],mb['m_structure_kg'],
              mb['m_avionics_kg'],mb['m_propellant_kg']]
    cols   = ['#d68910','#c0392b','#95a5a6','#2471a3','#7d3c98']
    wedges, _, autotexts = ax.pie(vals,colors=cols,autopct='%1.1f%%',
                                  startangle=90,pctdistance=0.75,
                                  wedgeprops={'edgecolor':'white','linewidth':2,'width':0.55})
    ax.set_title(f'Spacecraft Mass Breakdown\n(Wet mass: {mb["m_wet_kg"]:.0f} kg)')
    ax.legend(labels,loc='lower center',fontsize=7.5,ncol=2,
              bbox_to_anchor=(0.5,-0.12),framealpha=0.8)

    # Top-middle: specific power vs panel area
    ax = fig.add_subplot(gs[0,1])
    A_arr = mass_sens['values']
    sp    = np.array([r['specific_power_W_kg'] for r in mass_sens['results']])
    ax.semilogx(A_arr,sp,color='#1e8449',lw=2.5)
    ax.fill_between(A_arr,0,sp,alpha=0.1,color='#1e8449')
    ax.axvline(1e4,color='grey',ls=':',lw=1); ax.text(1.15e4,sp.max()*0.85,'10⁴ m²\nbaseline',fontsize=8,color='grey')
    ax.set_xlabel('Panel Area [m²]'); ax.set_ylabel('Specific Power [W/kg]')
    ax.set_title('Specific Power vs Panel Area\n(r=1 AU, η_PV=30%, Isp=3000 s)')

    # Top-right: TRL bar chart
    ax = fig.add_subplot(gs[0,2])
    sys_labels = [s.replace(' ','\n') for s in trl['subsystems']]
    trl_curr   = trl['TRL_current']
    trl_need   = trl['TRL_needed']
    y = np.arange(len(sys_labels))
    bars_cur = ax.barh(y,trl_curr,height=0.4,color='#2471a3',alpha=0.8,label='Current TRL')
    bars_gap = ax.barh(y,[n-c for n,c in zip(trl_need,trl_curr)],
                       height=0.4,left=trl_curr,color='#e8eaed',
                       edgecolor='#c0392b',linewidth=1.2,linestyle='--',label='TRL gap')
    ax.axvline(8,color='#c0392b',ls=':',lw=1.5); ax.text(8.05,len(y)-0.5,'TRL 8\n(needed)',fontsize=7.5,color='#c0392b')
    ax.set_yticks(y); ax.set_yticklabels(sys_labels,fontsize=7.5)
    ax.set_xlabel('Technology Readiness Level')
    ax.set_title('TRL by Sub-System')
    ax.legend(fontsize=8,loc='lower right')
    ax.set_xlim(0,9.5)

    # Bottom-left: radiator mass penalty
    ax = fig.add_subplot(gs[1,0])
    pv_rho = np.linspace(0.05,2.0,40)
    m_pv_arr = pv_rho*1e4
    from dyson_octahedron.thermal_management import waste_heat,radiator_area
    A_rad_1au = radiator_area(waste_heat(AU,1e4,0.30),400.0)
    m_rad_fix = A_rad_1au * 2.0
    m_tot = m_pv_arr + m_rad_fix + 0.15*(m_pv_arr+m_rad_fix) + 20
    ax.plot(pv_rho, m_tot, color='#7d3c98',lw=2.2)
    ax.axvline(0.5,color='grey',ls=':',lw=1); ax.text(0.52,m_tot.max()*0.9,'Current state\n0.5 kg/m²',fontsize=8,color='grey')
    ax.axvline(0.1,color='#1e8449',ls='--',lw=1.5); ax.text(0.02,m_tot.max()*0.6,'Target\n0.1 kg/m²',fontsize=8,color='#1e8449')
    ax.set_xlabel('PV Panel Areal Density [kg/m²]')
    ax.set_ylabel('Total Spacecraft Mass [kg]')
    ax.set_title('Mass vs Panel Areal Density\n(Radiator mass fixed at r=1 AU)')

    # Bottom-middle: mission lifetime propellant
    ax = fig.add_subplot(gs[1,1])
    years = np.linspace(1,30,50)
    from dyson_octahedron.station_keeping import annual_deltav, propellant_mass
    dv_yr = annual_deltav(AU,1e4,mb['m_dry_kg'])
    for isp,c_i in zip([450,1000,3000],['#d68910','#2471a3','#1e8449']):
        mp = np.array([propellant_mass(dv_yr,mb['m_dry_kg'],isp,y) for y in years])
        pf = mp/(mb['m_dry_kg']+mp)*100
        ax.plot(years,pf,color=c_i,lw=2.2,label=f'Isp={isp} s')
    ax.axhline(50,color='#c0392b',ls='--',lw=1.2,label='50% propellant cap')
    ax.set_xlabel('Mission Duration [years]')
    ax.set_ylabel('Propellant Fraction [%]')
    ax.set_title('Propellant Growth vs Mission Lifetime')
    ax.legend(fontsize=9)

    # Bottom-right: cost driver radar as bar
    ax = fig.add_subplot(gs[1,2])
    short_labels = ['Thin-film PV','Radiators','Ion prop.','Laser relay','ADCS','Autonomy','Assembly']
    mf = trl['mass_fraction']
    cd = trl['cost_driver']
    bar_c = ['#c0392b' if d else '#2471a3' for d in cd]
    ax.barh(short_labels, [m*100 for m in mf], color=bar_c, alpha=0.85,
            edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Mass Fraction [%]')
    ax.set_title('Sub-System Mass Share\n(■ Cost driver  ■ Non-critical)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#c0392b',label='Cost driver'),
                       Patch(color='#2471a3',label='Non-critical')],
              fontsize=8.5,loc='lower right')

    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Master Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def fig_dashboard(th, geo, mb, sk_sum, cov_map, net_f, save=None):
    fig = plt.figure(figsize=(18,12),facecolor='#f0f4f8')
    fig.suptitle('OctaDyson v2.0 — Complete System Dashboard  (r = 1 AU)',
                 fontsize=15,fontweight='bold',y=0.98)
    gs = gridspec.GridSpec(3,4,hspace=0.55,wspace=0.4,
                           top=0.93,bottom=0.05,left=0.06,right=0.97)

    # ── Power pie ──
    ax = fig.add_subplot(gs[0,0])
    Ph = th['P_harvest_total_W']; Pw = th['P_waste_total_W']
    ax.pie([Ph,Pw],labels=['Electrical','Waste Heat'],
           colors=['#1e8449','#c0392b'],autopct='%1.1f%%',
           startangle=90,wedgeprops={'edgecolor':'white','linewidth':2})
    ax.set_title(f'Power Budget (6 S/C)\n{Ph/1e6:.1f} MW elec. | {Pw/1e6:.1f} MW heat',fontsize=9)

    # ── Mass pie ──
    ax = fig.add_subplot(gs[0,1])
    mv=[mb['m_pv_kg'],mb['m_radiator_kg'],mb['m_structure_kg'],mb['m_avionics_kg'],mb['m_propellant_kg']]
    ax.pie(mv,colors=['#d68910','#c0392b','#95a5a6','#2471a3','#7d3c98'],
           autopct='%1.0f%%',startangle=45,pctdistance=0.75,
           wedgeprops={'edgecolor':'white','linewidth':1.5,'width':0.5})
    ax.set_title(f'Mass Budget\nWet: {mb["m_wet_kg"]:.0f} kg per S/C',fontsize=9)

    # ── Beam bar ──
    ax = fig.add_subplot(gs[0,2])
    geos_k = list(geo.keys()); geos_v = [v*100 for v in geo.values()]
    ax.barh(geos_k,geos_v,color=['#1e8449' if g=='Octahedron' else '#aab4c8' for g in geos_k],
            edgecolor='white',linewidth=1.2)
    ax.set_xlabel('η_beam [%]'); ax.set_xlim(0,90)
    ax.set_title('Beam Efficiency\nby Geometry',fontsize=9)

    # ── Key params table ──
    ax = fig.add_subplot(gs[0,3]); ax.axis('off')
    rows=[['Parameter','Value'],
          ['r_orbit','1.00 AU'],
          ['λ_max',  '0 (neutral)'],
          ['P_harvest/S/C',f'{th["P_harvest_each_W"]/1e6:.2f} MW'],
          ['P_waste/S/C',  f'{th["P_waste_each_W"]/1e6:.2f} MW'],
          ['A_rad/S/C',    f'{th["A_rad_each_m2"]:.0f} m²'],
          ['A_rad/A_panel',f'{th["rad_to_panel_ratio"]:.2f}'],
          ['ΔV annual',    f'{sk_sum["dv_annual_mps"]:.1f} m/s/yr'],
          ['m_wet/S/C',    f'{mb["m_wet_kg"]:.0f} kg'],
          ['Sky coverage', f'{cov_map["coverage_frac"]*100:.0f}% >30%'],
          ['η_oct edge',   f'{geo["Octahedron"]*100:.1f}%'],
          ['N spacecraft', '6'],
    ]
    tbl=ax.table(cellText=rows[1:],colLabels=rows[0],loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1,1.45)
    for (r_,c_),cell in tbl.get_celld().items():
        if r_==0: cell.set_facecolor('#1a3a6b'); cell.set_text_props(color='white',fontweight='bold')
        elif r_%2==0: cell.set_facecolor('#e8f4f0')
    ax.set_title('Key Parameters',fontsize=9,fontweight='bold',pad=10)

    # ── Coverage map ──
    ax = fig.add_subplot(gs[1,:2])
    eta = cov_map['eta_map']*100
    LON_d = np.degrees(cov_map['LON']); LAT_d = np.degrees(cov_map['LAT'])
    im=ax.pcolormesh(LON_d,LAT_d,eta,cmap='YlOrRd',vmin=0,vmax=100,shading='auto')
    plt.colorbar(im,ax=ax,label='η_beam [%]',shrink=0.8)
    ax.set_title(f'Sky Coverage Map — Mean η={cov_map["mean_eta"]*100:.1f}%',fontsize=9)
    ax.set_xlabel('Lon [°]'); ax.set_ylabel('Lat [°]')

    # ── Delta-V vs radius ──
    ax = fig.add_subplot(gs[1,2])
    from dyson_octahedron.station_keeping import deltav_vs_radius
    skd = deltav_vs_radius()
    ax.plot(skd['r_AU'],skd['dv_mps_yr'],color='#c0392b',lw=2)
    ax.axvline(1.0,color='grey',ls=':',lw=1)
    ax.set_xlabel('Orbital Radius [AU]'); ax.set_ylabel('Annual ΔV [m/s/yr]')
    ax.set_title('Station-Keeping ΔV',fontsize=9)

    # ── Thermal at 1 AU bar ──
    ax = fig.add_subplot(gs[1,3])
    cats=['P_harv\n[MW]','P_waste\n[MW]','A_rad\n[×100m²]']
    vals2=[th['P_harvest_each_W']/1e6,th['P_waste_each_W']/1e6,th['A_rad_each_m2']/100]
    ax.bar(range(3),vals2,color=['#d68910','#c0392b','#2471a3'],alpha=0.85,edgecolor='white',lw=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(cats,fontsize=8.5)
    for i,v in enumerate(vals2):
        ax.text(i,v+0.05,f'{v:.2f}',ha='center',fontsize=9,fontweight='bold')
    ax.set_title('Per-S/C Thermal Budget',fontsize=9)

    # ── SRP net force ──
    ax = fig.add_subplot(gs[2,0])
    pairlabels=['+x/−x','+y/−y','+z/−z']
    from dyson_octahedron.radiation_pressure import srp_all_vertices
    forces = srp_all_vertices(r=AU)
    net_pairs=[np.linalg.norm(forces[2*k]+forces[2*k+1]) for k in range(3)]
    ax.bar(pairlabels,net_pairs,color=C[:3],edgecolor='white',lw=1.5,alpha=0.85)
    ax.set_title('SRP Net Force per Pair\n(Should be ≈ 0)',fontsize=9)
    ax.set_ylabel('[N]')

    # ── TRL ──
    ax = fig.add_subplot(gs[2,1])
    from dyson_octahedron.mass_budget import technology_readiness
    trl2=technology_readiness()
    short=['PV','Radiator','Propulsion','Laser','ADCS','GNC','Assembly']
    y2=np.arange(len(short))
    ax.barh(y2,trl2['TRL_current'],color='#2471a3',alpha=0.8,height=0.5,label='Current')
    ax.barh(y2,[8-c for c in trl2['TRL_current']],left=trl2['TRL_current'],
            color='#e8eaed',edgecolor='#c0392b',lw=1,ls='--',height=0.5,label='Gap to TRL8')
    ax.axvline(8,color='#c0392b',ls=':',lw=1.2)
    ax.set_yticks(y2); ax.set_yticklabels(short,fontsize=8.5)
    ax.set_xlabel('TRL'); ax.set_title('Technology Readiness',fontsize=9)
    ax.legend(fontsize=7.5)

    # ── Propellant vs years ──
    ax = fig.add_subplot(gs[2,2])
    yrs=np.linspace(1,30,50)
    from dyson_octahedron.station_keeping import annual_deltav, propellant_mass
    dv_yr2=annual_deltav(AU,1e4,mb['m_dry_kg'])
    for isp2,c2 in zip([450,1000,3000],['#d68910','#2471a3','#1e8449']):
        mp2=np.array([propellant_mass(dv_yr2,mb['m_dry_kg'],isp2,y) for y in yrs])
        pf2=mp2/(mb['m_dry_kg']+mp2)*100
        ax.plot(yrs,pf2,color=c2,lw=2,label=f'Isp={isp2}s')
    ax.axhline(50,color='#c0392b',ls='--',lw=1)
    ax.set_xlabel('Mission Years'); ax.set_ylabel('Propellant %')
    ax.set_title('Propellant Fraction',fontsize=9); ax.legend(fontsize=8)

    # ── Geometry comparison table ──
    ax = fig.add_subplot(gs[2,3]); ax.axis('off')
    g5=[['','SRP','Stability','η_beam','Sky'],
        ['Oct','Exact','Neutral',f'{geo["Octahedron"]*100:.0f}%','Full'],
        ['Tet','Part','Marginal',f'{geo["Tetrahedron"]*100:.0f}%','No'],
        ['Cube','Part','Resonant',f'{geo["Cube"]*100:.0f}%','Part'],
        ['Ring','None','Planar',f'{geo["Ring (6-node)"]*100:.0f}%','Hem.'],
    ]
    gt=ax.table(cellText=g5[1:],colLabels=g5[0],loc='center',cellLoc='center')
    gt.auto_set_font_size(False); gt.set_fontsize(8); gt.scale(1.1,1.5)
    for (r_,c_),cell in gt.get_celld().items():
        if r_==0: cell.set_facecolor('#1a3a6b'); cell.set_text_props(color='white',fontweight='bold')
        if r_==1 and c_>0: cell.set_facecolor('#d5f5e3')
    ax.set_title('Geometry Comparison',fontsize=9,fontweight='bold',pad=8)

    if save: plt.savefig(save,dpi=160,bbox_inches='tight'); plt.close()
    return fig
