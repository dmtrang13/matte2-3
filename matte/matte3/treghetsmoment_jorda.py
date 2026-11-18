import numpy as np
import math
from scipy.integrate import tplquad

R = 6371000  # Radius til jorden

def rho(r):
    # Jordens massetetthet som funksjon av radiell distanse fra jordsenteret
    # Modell fra: World Journal of Condensed Matter Physics, 2021, 11, 13-27
    r_hat = (r - 0.5) / R
    if r <= int(0.1915 * R):
        return -8088.796304 * r_hat**2 - 330.8910692 * r_hat + 13100
    elif r <= int(0.5469 * R):
        return -10693.43214 * r_hat**2 + 1635.478802 * r_hat + 12203.9503
    elif r <= int(0.8947 * R):
        return -3450.258769 * r_hat + 7486.946521
    elif r <= int(0.9953 * R):
        return -6212.723658 * r_hat + 9558.523857
    elif r <= int(0.9992 * R):
        return -102564.1026 * r_hat + 105082.0513
    else:
        return -125000 * r_hat + 126000

def integrand_volum(r, theta, phi):
    return r**2 * math.sin(phi); # Din kode for integranden for utregningen av jordens volum i m^3

def integrand_masse(r, theta, phi):
    return rho(r) * r**2 * math.sin(phi); # Din kode for integranden for utregningen av jordens masse i kg

def integrand_treghetsmoment(r, theta, phi):
    return rho(r) * r**4 * (math.sin(phi)**3);# Din kode for integranden for utregningen av jordens treghetsmoment i kg m^2

volum, error_volum = tplquad(
    integrand_volum,  # integrand
    0, np.pi,         # integrasjonsgrenser for phi
    0, 2*np.pi,       # integrasjonsgrenser for theta
    0, R,             # integrasjonsgrenser for r
    epsrel=1e-4       # Toleranse for relativ feil i integralet
)

masse, error_masse = tplquad(
    integrand_masse, 
    0, np.pi,         # integrasjonsgrenser for phi
    0, 2*np.pi,       # integrasjonsgrenser for theta
    0, R,             # integrasjonsgrenser for r
    epsrel=1e-4       # Toleranse for relativ feil i integralet
)

treghetsmoment, error_treghetsmoment = tplquad(
    integrand_treghetsmoment,
    0, np.pi,      # integrasjonsgrenser for phi
    0, 2*np.pi,    # integrasjonsgrenser for theta
    0, R,          # integrasjonsgrenser for r
    epsrel=1e-4    # Toleranse for relativ feil i integralet
)

