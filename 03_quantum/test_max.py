import numpy as np
from scipy.integrate import solve_ivp

L0 = 1.0
L_end = 10.0
v_fast = 40.0
N_bas = 12
hbar = 1.0
m = 1.0
T_fast = (L_end - L0) / v_fast
t_fast = np.linspace(0, T_fast, 500)

def M_nm(n, m, L):
    if n == m: return 0.0
    if (n + m) % 2 == 0: return 0.0
    return -(2.0 * n * m) / (L * (n**2 - m**2))

def build_ODE(v_exp):
    def rhs(t, c):
        L  = L0 + v_exp * t
        dc = np.zeros(N_bas, dtype=complex)
        E  = np.array([(n+1)**2 * np.pi**2 * hbar**2 / (2*m*L**2) for n in range(N_bas)])
        for i in range(N_bas):
            dc[i] = -1j * (E[i] / hbar) * c[i]
            for j in range(N_bas):
                if j != i:
                    Mij = M_nm(i+1, j+1, L)
                    if Mij != 0.0:
                        dc[i] -= v_exp * Mij * c[j]
        return dc
    return rhs

c0 = np.zeros(N_bas, dtype=complex)
c0[0] = 1.0 + 0j
sol_fast = solve_ivp(build_ODE(v_fast), [0, T_fast], c0, t_eval=t_fast, method='DOP853', atol=1e-9, rtol=1e-9)

max_d = 0
x_res = 600
for frame in range(500):
    L_f = L0 + v_fast * t_fast[frame]
    c_f = sol_fast.y[:, frame]
    x_f = np.linspace(0, L_f, x_res)
    phi_f = np.array([np.sqrt(2/L_f) * np.sin((n+1)*np.pi*x_f/L_f) for n in range(N_bas)])
    psi_f = np.einsum('n,nx->x', c_f, phi_f)
    density_f = np.abs(psi_f)**2
    if np.max(density_f) > max_d:
        max_d = np.max(density_f)
print(f"Max density: {max_d}")
