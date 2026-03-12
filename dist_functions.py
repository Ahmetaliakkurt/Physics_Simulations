import numpy as np
import matplotlib.pyplot as plt

k_B = 8.617333262145e-5
T = 300
E_f = 1.0 

def fermi_dirac(E, mu, T):
    return 1 / (np.exp((E - mu) / (k_B * T)) + 1)

def bose_einstein(E, mu, T):
    with np.errstate(divide='ignore', invalid='ignore'):
        val = 1 / (np.exp((E - mu) / (k_B * T)) - 1)
        val[E <= mu] = np.inf
    return val

def maxwell_boltzmann(E, T):
    return np.exp(-E / (k_B * T))

E = np.linspace(0.001, 2.5, 500)
x_axis = E / E_f

fd = fermi_dirac(E, E_f, T)
be = bose_einstein(E, 0.0, T)
mb = maxwell_boltzmann(E, T)
mb = mb / np.max(mb)

plt.figure(figsize=(10, 6))

plt.plot(x_axis, fd, label='Fermi-Dirac', linewidth=2)
plt.plot(x_axis, be, label='Bose-Einstein', linewidth=2, linestyle='--')
plt.plot(x_axis, mb, label='Maxwell-Boltzmann', linewidth=2, linestyle='-.')

plt.xlabel('$E / E_f$')
plt.ylabel('Occupation Probability')
plt.title(f'Comparison of Statistics at $T= {T}K$')
plt.ylim(0, 1.2)
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()