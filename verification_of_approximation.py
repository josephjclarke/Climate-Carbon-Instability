#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from conceptual_model_with_fossil_carbon import ClimateCarbonSystem

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

Alk = 5130.0
Ca05 = 344.0 * 2.12
CF = 2 * 2.12
Ca0 = 286.085
CL0 = 1630.0
npp0 = 65.0

chalf = np.linspace(0.0, 2500.0, 250)
ECS = np.linspace(5.0, 15.0, 250)
Ca0s = np.array([190, Ca0, 2 * Ca0])

stability = np.zeros((Ca0s.size, chalf.size, ECS.size))
critical_ECS = np.zeros((Ca0s.size, chalf.size))

for i, j, k in np.ndindex(stability.shape):
    ccs = ClimateCarbonSystem(
        Alk=Alk,
        Ca05=chalf[j],
        Ca0=Ca0s[i] * 2.12,
        CL0=CL0,
        npp0=npp0,
        L=3.7 / ECS[k],
    )
    stability[i, j, k] = ccs.jac_eig_at(ccs.pi_equilibrium).real.max()
    critical_ECS[i, j] = ccs.analytic_bf(use_simple_approx=True)[1]

da = xr.DataArray(stability, coords={"Ca0": Ca0s, "chalf": chalf, "ECS": ECS})

g = da.plot(
    col="Ca0",
    x="chalf",
    y="ECS",
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    vmin=-0.002,
    vmax=0.002,
    center=0.0,
    xlim=(0.0, chalf.max()),
)

ticks = g.cbar.get_ticks()
tick_labels = [str(t) for t in ticks]
tick_labels[0] = "Stable"
tick_labels[-1] = "Unstable"
g.cbar.set_ticks(ticks, labels=tick_labels)
g.cbar.set_label(r"Growth Rate (yr$^{-1}$)")

for i in range(len(Ca0s)):
    ax = g.axs.flatten()[i]
    ax.set_xlabel(r"$C_{1/2}$ (ppmv)")
    ax.plot(chalf, critical_ECS[i], color="black", label="Analytic Estimate")
    if i == 0:
        ax.set_ylabel("ECS (K)")
    if i == 2:
        ax.legend(frameon=False)
    if i == 0:
        ax.set_title("LGM")
    if i == 1:
        ax.set_title("Pre-Industrial")
    if i == 2:
        ax.set_title(r"$2 \times $CO$_2$")

plt.savefig("figures/verification_of_approximation.pdf")
plt.close()
