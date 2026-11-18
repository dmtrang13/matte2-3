import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Verdier fra oppgave 3
R = 10.0
L = 5e-3
C = 50e-6

omega_0 = 1/np.sqrt(L*C)
zeta = (R/2)*np.sqrt(C/L)

# Transferfunksjon: H(s) = ω0^2 / (s^2 + 2ζω0 s + ω0^2)
num = [omega_0**2]
den = [1, 2*zeta*omega_0, omega_0**2]
system = sig.TransferFunction(num, den)

print(f"ω0 = {omega_0:.2f} rad/s, ζ = {zeta:.3f}")

# Funksjoner
def estimate_amp_phase(t, x, y, omega, tail_cycles=2):
    T = 2*np.pi/omega
    t0 = t[-1] - tail_cycles*T
    m = t >= t0
    tt, xx, yy = t[m], x[m], y[m]

    Ax = 0.5*(xx.max() - xx.min())
    Ay = 0.5*(yy.max() - yy.min())
    mag = Ay/Ax

    def rising_zero(tt, zz):
        s = np.sign(zz)
        idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
        i = idx[0]
        t1, t2 = tt[i], tt[i+1]
        z1, z2 = zz[i], zz[i+1]
        return t1 + (0 - z1)*(t2 - t1)/(z2 - z1)

    tx = rising_zero(tt, xx)
    ty = rising_zero(tt, yy)
    dt = ty - tx
    ph = -omega*dt
    ph = (ph + np.pi) % (2*np.pi) - np.pi
    return mag, dt, ph

def H_jw(omega):
    s = 1j*omega
    return (omega_0**2) / (s**2 + 2*zeta*omega_0*s + omega_0**2)

# 4a
omega = 4000.0
t = np.linspace(0, 0.02, 8000)
v0 = np.sin(omega*t)
_, vC, _ = sig.lsim(system, U=v0, T=t)

plt.figure()
plt.plot(t, v0, label=f"v0(t)=sin({omega}t)")
plt.plot(t, vC, label="vC(t)")
plt.grid(True)
plt.xlabel("t [s]")
plt.ylabel("U [V]")
plt.legend()
plt.title("RLC: input og kondensatorspenning")
plt.show()

T = 2*np.pi/omega
plt.figure()
plt.plot(t, v0, label="v0(t)")
plt.plot(t, vC, label="vC(t)")
plt.xlim(t[-1] - 2*T, t[-1])
plt.grid(True)
plt.xlabel("t [s]")
plt.ylabel("U [V]")
plt.legend()
plt.title("Zoom: siste periode (ω=500)")
plt.show()

#4b + 4c
print("\n   ω | |H|_målt    Δt [s]        ∠_målt [rad]   ∠_målt [deg]  ||  |H|_teori   ∠_teori [rad]   ∠_teori [deg]")
print("-----+------------------------------------------------------------------------------------------------------")

for omega in [500.0, 1000.0, 2000.0, 4000.0]:
    t = np.linspace(0, 0.02, 8000)
    v0 = np.sin(omega*t)
    _, vC, _ = sig.lsim(system, U=v0, T=t)

    mag_meas, dt, ph_meas = estimate_amp_phase(t, v0, vC, omega, tail_cycles=2)

    H = H_jw(omega)
    mag_th = np.abs(H)
    ph_th = np.angle(H)
    ph_th = (ph_th + np.pi) % (2*np.pi) - np.pi

    print(f"{omega:4.0f} | {mag_meas:8.4f}  {dt: .6e}    "
          f"{ph_meas:10.4f}      {np.degrees(ph_meas):10.2f}  ||  "
          f"{mag_th:8.4f}     {ph_th:10.4f}      {np.degrees(ph_th):10.2f}")