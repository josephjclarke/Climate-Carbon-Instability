#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from conceptual_model_with_fossil_carbon import ClimateCarbonSystem

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

Ca0 = 286.085
Ca05 = 344.0 * 2.12
Alk = 5130.0
CL0 = 1630.0
npp0 = 65.0


@np.vectorize
def find_critical_ecs_ca0_chalf(ca0, chalf, q10=2.0):
    def f(ECS):
        ccs = ClimateCarbonSystem(
            Alk=Alk,
            Ca05=chalf,
            Ca0=ca0 * 2.12,
            CL0=CL0,
            npp0=npp0,
            L=3.7 / ECS,
            Q10=q10,
        )
        return ccs.jac_eig_at(ccs.pi_equilibrium).real.max()

    return scipy.optimize.root_scalar(f, method="bisect", bracket=(0.1, 1e2)).root


print("Should be near 10.9: ", find_critical_ecs_ca0_chalf(Ca0, Ca05))

ch = np.linspace(0.0, 600.0, 100)
ca = np.linspace(100.0, 2 * Ca0, 100)

ch_mesh, ca_mesh = np.meshgrid(ch, ca)

crit_ecs_chalf_cas_2 = find_critical_ecs_ca0_chalf(ca_mesh, ch_mesh * 2.12, q10=2.0)
crit_ecs_chalf_cas_3 = find_critical_ecs_ca0_chalf(ca_mesh, ch_mesh * 2.12, q10=3.0)


levels = np.linspace(0.0, 14.0, 100)
contour_levels = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
cmap = "viridis"

fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2, sharey=True, sharex=True, figsize=(12.5, 5)
)
cs1 = ax1.contourf(ch, ca, crit_ecs_chalf_cas_2, levels=levels, extend="max", cmap=cmap)
cs2 = ax2.contourf(
    ch,
    ca,
    crit_ecs_chalf_cas_3,
    levels=levels,
    extend="max",
    cmap=cmap,
)
for c in cs1.collections:
    c.set_rasterized(True)
for c in cs2.collections:
    c.set_rasterized(True)
ct1 = ax1.contour(ch, ca, crit_ecs_chalf_cas_2, levels=contour_levels, colors="black")
ct2 = ax2.contour(
    ch,
    ca,
    crit_ecs_chalf_cas_3,
    levels=sorted(contour_levels + [3.0, 5.0, 7.0]),
    colors="black",
)

cbar = fig.colorbar(cs1, ax=[ax1, ax2], ticks=np.arange(0.0, 16.0, 2.0))
cbar.set_label("Critical ECS (K)")

fmt = lambda x: f"{x:.0f}K"
ax1.clabel(ct1, fmt=fmt, inline=True)
ax2.clabel(ct2, fmt=fmt, inline=True)

fig.supxlabel(r"$C_{1/2}$ (ppm)", x=0.45)
fig.supylabel(r"$C_{A}^*$ (ppm)", x=0.05)
ax1.set_title(r"$Q_{10} = 2$")
ax2.set_title(r"$Q_{10} = 3$")
plt.savefig("figures/critical_ecs_as_func_of_q10_chalf_ca0.pdf")
plt.close()
