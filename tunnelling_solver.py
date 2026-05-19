import numpy as np 
import matplotlib.pyplot as plt 
# --- Parameters --- 
L_left = -6.0 
L_right = 8.0 
num_points = 12000 
# Tuned for strong but not total transmission 
U0 = 25.0     
 # barrier height 
E = 24.0       
a = 0.0 
b = 1.5      
# energy close to U0 (E < U0 but nearly equal) 
  # barrier width 
# --- Constants --- 
hbar = 1.0 
m = 1.0 
# --- Spatial domain --- 
x = np.linspace(L_left, L_right, num_points) 
dx = x[1] - x[0] 
# --- Wavenumbers --- 
k = np.sqrt(2 * m * E) / hbar 
kappa = np.sqrt(2 * m * (U0 - E)) / hbar 
ik = 1j * k 
# --- Solve for coefficients --- 
w = b - a 
M = np.array([ 
[1, -1, -1, 0], 
[-ik, kappa, -kappa, 0], 
[0, np.exp(-kappa * w), np.exp(kappa * w), -1], 
[0, -kappa * np.exp(-kappa * w), kappa * np.exp(kappa * w), -ik] 
], dtype=complex) 
y = np.array([-1, -ik, 0, 0], dtype=complex) 
v = np.linalg.solve(M, y) 
R, C, D, F = v 
print(f"--- Final Tuned Parameters ---") 
print(f"Barrier height U₀ = {U0}, Energy E = {E}, Width = {b - a}") 
print(f"R (Reflection Amp) = {R:.4f}") 
print(f"F (Transmission Amp) = {F:.4f}") 
T_prob = np.abs(F)**2 
R_prob = np.abs(R)**2 
print(f"Transmission Probability |F|² = {T_prob:.4f}") 
print(f"Reflection Probability |R|² = {R_prob:.4f}") 
print(f"Total Probability |R|² + |F|² = {R_prob + T_prob:.4f}") 
# --- Wavefunction --- 
psi = np.zeros_like(x, dtype=complex) 
region1 = x < a 
psi[region1] = np.exp(1j * k * (x[region1] - a)) + R * np.exp(-1j * k * (x[region1] - a)) 
region2 = (x >= a) & (x <= b) 
psi[region2] = C * np.exp(-kappa * (x[region2] - a)) + D * np.exp(kappa * (x[region2] - a)) 
region3 = x > b 
psi[region3] = F * np.exp(1j * k * (x[region3] - b)) 
# --- Potential --- 
V = np.zeros_like(x) 
V[region2] = U0 
# --- Plot --- 
plt.figure(figsize=(12, 7)) 
max_psi_mag = np.max(np.abs(psi)) 
plot_y_max = max_psi_mag * 1.5 
plt.plot(x, np.real(psi), 'b', linewidth=2, label='Re(ψ(x))') 
plt.plot(x, np.imag(psi), 'g', linewidth=1, linestyle=':', label='Im(ψ(x))') 
# Scaled potential and energy lines 
V_plot_height = plot_y_max * 0.8 
plot_V = V / U0 * V_plot_height 
plot_E = (E / U0) * V_plot_height 
plt.plot(x, plot_V, 'r--', linewidth=2, label=f'Potential Barrier (scaled, U₀={U0})') 
plt.axhline(plot_E, color='orange', linestyle='-.', linewidth=2, label=f'Particle Energy (scaled, E={E})') 
plt.axvline(a, color='gray', linestyle='--', linewidth=1) 
plt.axvline(b, color='gray', linestyle='--', linewidth=1) 
plt.xlim(L_left, L_right) 
plt.ylim(-plot_y_max, plot_y_max) 
plt.xlabel('x (position)', fontsize=12) 
plt.ylabel('ψ(x)', fontsize=12) 
plt.title(f'Quantum Tunneling (E={E} < U₀={U0}, width={b-a}): Continuous Transmission', fontsize=14) 
plt.legend(loc='upper right', fontsize=10) 
plt.grid(True, linestyle=':') 
y_text = plot_y_max * 0.9 
plt.text((L_left + a)/2, y_text, 'REGION I\nIncident + Reflected', ha='center') 
plt.text((a + b)/2, y_text, 'REGION II\nBarrier', ha='center') 
plt.text((b + L_right)/2, y_text, 'REGION III\nTransmitted', ha='center') 
plt.text((a + b)/2, -plot_y_max * 0.5, r'$e^{-\kappa x}$', fontsize=14, color='purple', ha='center') 
plt.tight_layout() 
plt.savefig('quantum_tunneling_visible_transmission.png') 
plt.show()