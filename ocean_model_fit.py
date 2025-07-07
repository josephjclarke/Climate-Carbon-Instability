#!/usr/bin/env python3
"""
Created on Tue Aug 20 11:14:01 2024.

@author: jc1147
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
import numdifftools as nd

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


k1 = 1e-6
k2 = 7.53e-10

gcb = np.genfromtxt("data/gcb/gcb.csv", delimiter=",", names=True, skip_header=15)


years = gcb["Year"]
fo = gcb["ocean_sink"]
ca = np.nancumsum(gcb["atmospheric_growth"]) + 602.769

co2 = scipy.interpolate.CubicSpline(years, ca)


def H(C1, k, Alk):
    p0 = 1
    p1 = k1 - C1 * k1 / Alk
    p2 = (1 - 2 * C1 / Alk) * k1 * k2
    return 1 / (np.roots([p2, p1, p0])).max()


def E(C1, k, Alk):
    h = H(C1, k, Alk)
    return k * C1 / (1 + k1 / h + k1 * k2 / h**2)


def C1eq(k, Alk):
    return scipy.optimize.fsolve(lambda C1: ca[0] - E(C1, k, Alk), x0=[1000.0])


def pH(k, Alk):
    C1 = C1eq(k, Alk)
    return -np.log10(H(C1, k, Alk))


def Eprime(C1, k, Alk):
    return nd.Derivative(lambda x: E(x, k, Alk))(C1)


Evec = np.vectorize(E)


def model(t, y, nu1, nu2, V1V2, k, Alk):
    C1, C2 = y
    dC1 = nu1 * (co2(t) - E(C1, k, Alk)) - nu2 * (C1 - V1V2 * C2)
    dC2 = nu2 * (C1 - V1V2 * C2)
    return [dC1, dC2]


def jac(t, y, nu1, nu2, V1V2, k, Alk):
    C1, C2 = y
    return [[-nu1 * Eprime(C1, k, Alk) - nu2, V1V2 * nu2], [nu2, -nu2 * V1V2]]


def solve_model(nu1, nu2, V1V2, k, Alk, forward=False):
    C1 = C1eq(k, Alk)
    C2 = C1 / V1V2
    C1s = np.zeros_like(years)
    C2s = np.zeros_like(years)
    C1s[0] = C1
    C2s[0] = C2
    if forward:
        for i in range(1, years.size):
            dC1, dC2 = model(years[i], [C1s[i - 1], C2s[i - 1]], nu1, nu2, V1V2, k, Alk)
            C1s[i] = C1s[i - 1] + dC1
            C2s[i] = C2s[i - 1] + dC2
    else:
        for i in range(1, years.size):
            xm1 = np.array([C1s[i - 1], C2s[i - 1]])
            C1s[i], C2s[i] = scipy.optimize.fsolve(
                lambda x: x - model(years[i], x, nu1, nu2, V1V2, k, Alk) - xm1,
                xm1,
                fprime=lambda x: np.eye(2) - jac(years[i], x, nu1, nu2, V1V2, k, Alk),
            )
    return C1s, C2s


def calculate_flux(nu1, nu2, V1V2, k, Alk):
    C1s, C2s = solve_model(nu1, nu2, V1V2, k, Alk)
    return np.ediff1d(C1s + C2s, to_begin=0.0)


nu1_init = 0.2
nu2_init = 0.002
V1V2_init = 1 / 50.0
k_init = 220.0
Alk_init = 662.7
p0 = [nu1_init, nu2_init, V1V2_init, k_init, Alk_init]

popt, pcov = scipy.optimize.curve_fit(
    lambda x, Alk: calculate_flux(nu1_init, nu2_init, V1V2_init, k_init, Alk)[31:],
    co2(years)[31:],
    fo[31:],
    p0=[Alk_init * 7],
    bounds=(0.0, np.inf),
    x_scale=[Alk_init],
    sigma=np.full_like(fo[31:], 0.4),
)
fitted_Alk = popt.item()

plt.plot(years, fo, label="GCB")
plt.fill_between(years, fo - 0.4, fo + 0.4, alpha=0.1)
plt.plot(
    years,
    calculate_flux(nu1_init, nu2_init, V1V2_init, k_init, fitted_Alk),
    label="Model",
)
plt.xlabel("Time")
plt.ylabel("Ocean Uptake PgC/yr")
plt.title("Ocean Carbon Uptake")
plt.legend(frameon=False)
plt.savefig("figures/alk_fit.pdf")
plt.close()

print("Fitting Alk")
print("p0", p0)
print("popt", popt)
print("pH", pH(k_init, fitted_Alk))
print("E'(C1)", Eprime(C1eq(k_init, fitted_Alk), k_init, fitted_Alk).item())
