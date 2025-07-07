#!/usr/bin/env python3
"""
Created on Thu Oct 24 12:06:31 2024.

@author: jc1147
"""

import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


def cosdeg(x):
    """Calculate the cosine of argument in degrees."""
    return np.cos(np.deg2rad(x))


earth_radius = 6400.0 * 1000.0
secs_in_year = 360 * 24 * 60 * 60.0

dA = earth_radius**2 * np.deg2rad(3.75) * np.deg2rad(2.5)


files = sorted(glob.glob("data/jules_control/vn5.1_imogen_noff.spinup_*.dump.*.0.nc"))

data = xr.concat([xr.open_dataset(file) for file in files], "time").assign(
    time=10 * np.arange(len(files))
)

lats = data.latitude.isel(time=0)
areas = dA * cosdeg(lats)

cs = (data["cs"].sum(["sclayer", "scpool"]) * areas).sum("land") * 1e-12

cs_fit = scipy.stats.linregress(cs.time, cs)

fig, ax = plt.subplots(nrows=1, ncols=1)
cs.plot(ax=ax)
ax.set_ylim(1620.0, 1640.0)
ax.set_xlim(data.time.min(), data.time.max())
ax.set_xlabel("Time (years)")
ax.set_ylabel("Soil Carbon (PgC)")
plt.savefig("figures/soil_carbon_equilibrium.pdf")
plt.close()

print(f"Soil carbon is changing at a rate of {cs_fit.slope:.1g} PgC/yr")
print(
    "This represents a change of"
    + f" {100 * data.time[-1].item()/cs[0] * cs_fit.slope:.1g}%"
    + " over the simulation"
)
