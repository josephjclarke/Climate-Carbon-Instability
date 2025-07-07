#!/usr/bin/env python3
"""
Created on Thu Apr 18 11:39:00 2024.

@author: jc1147
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats

earth_radius = 6400.0 * 1000.0
secs_in_year = 360 * 24 * 60.0 * 60.0

Ca0ppm = 286.085

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


def cosdeg(x):
    """Calculate the cosine of argument in degrees."""
    return np.cos(np.deg2rad(x))


def nppfit(x, pi0, xhalf):
    """Fit to NPP."""
    return pi0 * x * (1 + xhalf) / (x + xhalf)


data = xr.open_mfdataset("data/jules_forced/*.nc").compute()

lats = data.latitude.isel(y=0)
areas = earth_radius**2 * cosdeg(lats) * np.deg2rad(3.75) * np.deg2rad(2.5)
co2_ppm = 28.9644 / 44.0095 * 1e6 * data["co2_mmr"].isel(x=0, y=0)
npp = (data["npp_gb"].isel(y=0) * areas).sum("x") * secs_in_year * 1e-12

fit = scipy.stats.linregress(Ca0ppm / co2_ppm, npp[0] / npp)
fitted_xhalf = fit.slope / (1 - fit.slope)

fig, ax = plt.subplots()
ax.plot(co2_ppm, npp, color="black", label="JULES", linestyle=":")
ax.plot(
    co2_ppm,
    nppfit(co2_ppm / Ca0ppm, npp[0], fitted_xhalf),
    label=r"$\Pi_0 = 65$PgC, $C_{1/2} = $" + f" {fitted_xhalf * Ca0ppm:.0f}ppm",
)
ax.set_xlabel("CO2 ppm")
ax.set_ylabel("NPP PgC/yr")
ax.legend(frameon=False)
plt.savefig("figures/jules_co2_npp.pdf")
plt.close()
