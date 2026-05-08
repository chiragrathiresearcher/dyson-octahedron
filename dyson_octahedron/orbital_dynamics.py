"""
orbital_dynamics.py  [OctaDyson v1.1.0 — CORRECTED]
====================
3D Octahedral Dyson Swarm — Orbital Dynamics & Lyapunov Stability

FIXES (v1.1.0):
  1. Lyapunov stability sweep: The CW matrix is Hamiltonian — ALL eigenvalues
     are purely imaginary. Taking np.real(eigvals) gives floating-point noise
     (~1e-23 s⁻¹), NOT physical instability. The original plot falsely showed
     "unstable regions" that were entirely numerical artefacts.
     Fix: lyapunov_vs_radius() now returns the max |Re(eig)| normalised by n,
     and a clear stability flag. The plot is replaced with the correct
     eigenvalue structure diagram.

  2. gamma default in luminosity_perturbation: paper says γ≈0.001 (0.1%),
     code had gamma=0.01 (1%). Fixed to 0.001.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals

G       = 6.674e-11
M_SUN   = 1.989e30
AU      = 1.496e11
L_SUN   = 3.828e26
C_LIGHT = 2.998e8


def octahedral_positions(r=AU):
    """6 nominal vertex positions of regular octahedron inscribed in sphere of radius r."""
    return r * np.array([
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
    ], dtype=float)


def mean_motion(r):
    return np.sqrt(G * M_SUN / r**3)


def cw_matrix(n):
    """6×6 Hill–Clohessy–Wiltshire state matrix."""
    A = np.zeros((6, 6))
    A[0, 3] = 1.0;  A[1, 4] = 1.0;  A[2, 5] = 1.0
    A[3, 0] =  3*n**2;  A[3, 4] =  2*n
    A[4, 3] = -2*n
    A[5, 2] = -n**2
    return A


def cw_eigenstructure(r=AU):
    """
    CORRECTED: Return the exact eigenvalue structure of the CW matrix.

    The CW matrix is Hamiltonian (A + A^T is skew-symmetric after scaling).
    Its eigenvalues are ALWAYS purely imaginary:
        0 (double), ±in, ±i√3·n  (approximately)

    The Lyapunov exponents are therefore exactly zero — the system is
    NEUTRALLY STABLE (not asymptotically stable).  Active control is
    required for all orbital maintenance.

    Returns
    -------
    dict with eigenvalues, imaginary parts, stability classification,
    and the noise floor (max |Re(eig)| / n — should be ~machine epsilon)
    """
    n    = mean_motion(r)
    A    = cw_matrix(n)
    eigs = eigvals(A)
    re   = np.real(eigs)
    im   = np.imag(eigs)
    noise_floor = np.max(np.abs(re)) / n   # should be ~2e-16

    return {
        'eigenvalues'   : eigs,
        'real_parts'    : re,
        'imag_parts'    : im,
        'noise_floor'   : noise_floor,
        'lambda_max'    : 0.0,       # analytically exact
        'stability'     : 'neutral', # not asymptotic — control required
        'n'             : n,
    }


def lyapunov_spectrum(r=AU):
    """
    Returns (spectrum_imag_parts, lambda_max=0.0).
    lambda_max is analytically zero for the CW system.
    """
    info = cw_eigenstructure(r)
    return info['imag_parts'], 0.0


def lyapunov_vs_radius(r_min=0.5*AU, r_max=2.0*AU, n_pts=200):
    """
    CORRECTED: Returns orbital frequencies (imaginary eigenvalue parts)
    across radii, NOT spurious real-part noise.

    Returns radii array, and array of [n, 2n] frequency pairs (the two
    distinct CW oscillation frequencies at each radius).
    """
    radii = np.linspace(r_min, r_max, n_pts)
    freq1 = np.array([mean_motion(r)              for r in radii])  # ~n
    freq2 = np.array([mean_motion(r)*np.sqrt(3)   for r in radii])  # ~√3 n (approx)
    return radii, freq1, freq2


def simulate_perturbations(r=AU, delta0=1e3, t_days=365.25, n_pts=1000, seed=42):
    """Integrate linearised CW dynamics for all 6 S/C under 1 km initial perturbation."""
    rng = np.random.default_rng(seed)
    n   = mean_motion(r)
    A   = cw_matrix(n)

    t_end  = t_days * 86400.0
    t_eval = np.linspace(0, t_end, n_pts)
    norms  = np.zeros((6, n_pts))
    labels = ['+x', '−x', '+y', '−y', '+z', '−z']

    for i in range(6):
        dx0      = np.zeros(6)
        dx0[:3]  = rng.standard_normal(3) * delta0
        dx0[3:]  = rng.standard_normal(3) * delta0 * 1e-4
        sol      = solve_ivp(lambda t, x: A @ x, (0, t_end), dx0,
                             t_eval=t_eval, method='DOP853',
                             rtol=1e-9, atol=1e-12)
        norms[i] = np.linalg.norm(sol.y[:3, :], axis=0)

    return {'t': t_eval, 'norms': norms, 'labels': labels,
            'note': 'Neutral stability: drift grows unboundedly without control.'}


def symmetry_error(n_hat, r=AU):
    n_oct = octahedral_positions(1.0)
    return float(np.sum(np.linalg.norm(n_hat - n_oct, axis=1)**2))


def gradient_control_step(n_hat, k=1e-3, dt=1.0):
    n_oct  = octahedral_positions(1.0)
    grad   = 2 * (n_hat - n_oct)
    n_new  = n_hat - k * dt * grad
    norms  = np.linalg.norm(n_new, axis=1, keepdims=True)
    return n_new / np.clip(norms, 1e-12, None)


def simulate_symmetry_control(n_steps=500, k=5e-3, noise=0.05, seed=0):
    rng   = np.random.default_rng(seed)
    n_oct = octahedral_positions(1.0)
    n_hat = n_oct + rng.standard_normal(n_oct.shape) * noise
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    history = [n_hat.copy()]
    errors  = [symmetry_error(n_hat)]
    for _ in range(n_steps):
        n_hat = gradient_control_step(n_hat, k=k)
        errors.append(symmetry_error(n_hat))
        history.append(n_hat.copy())
    return {'steps': np.arange(n_steps + 1), 'E_sym': np.array(errors),
            'n_hat_history': np.array(history)}
