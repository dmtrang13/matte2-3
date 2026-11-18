import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

R = 10.0
L = 5e-3
C = 50e-6

omega_0 = 1/np.sqrt(L*C)
zeta = (R/2)*np.sqrt(C/L)

num = [omega_0**2]
den = [1, 2*zeta*omega_0, omega_0**2]
system = sig.TransferFunction(num, den)

w = np.logspace(2, np.log10(4000), 1000)

w_out, mag_db, phase_deg = sig.bode(system, w=w)

targets = np.array([500.0, 1000.0, 2000.0, 4000.0])
_, mag_db_pts, phase_deg_pts = sig.bode(system, w=targets)

# Amplitude
plt.figure()
plt.semilogx(w_out, mag_db)
plt.semilogx(targets, mag_db_pts, 'ro')
plt.xlim(100, 4000)
plt.grid(True, which="both")
plt.xlabel(r"$\omega$ [rad/s]")
plt.ylabel(r"$20\log_{10}|H(j\omega)|$ [dB]")
plt.title("Bode-plot: amplitude")
plt.show()

# Fase
plt.figure()
plt.semilogx(w_out, phase_deg)
plt.semilogx(targets, phase_deg_pts, 'ro')
plt.xlim(100, 4000)
plt.grid(True, which="both")
plt.xlabel(r"$\omega$ [rad/s]")
plt.ylabel(r"$\angle H(j\omega)$ [deg]")
plt.title("Bode-plot: fase")
plt.show()

# Konverter til lineær amplitude
# Hdb = 20*log10(H) => H = 10**(Hdb/20)
mag_lin_pts = 10**(mag_db_pts/20)

print("\nVerdier lest fra Bode-plot:\n")
print(" ω (rad/s) |   |H| (lin)   |   dB       |   Fase (deg)")
print("--------------------------------------------------------")

for w0, mlin, mdb, ph in zip(targets, mag_lin_pts, mag_db_pts, phase_deg_pts):
    print(f"{w0:8.0f}   |   {mlin:8.4f}    |  {mdb:7.2f}   |   {ph:8.2f}")