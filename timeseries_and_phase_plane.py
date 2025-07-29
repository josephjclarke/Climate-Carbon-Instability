import matplotlib.pyplot as plt
import pandas as pd

from conceptual_model_with_fossil_carbon import ClimateCarbonSystem

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

Alk = 5130.0
Ca05 = 344.0 * 2.12
CF = 20 * 2.12

Ca0 = 286.085
CL0 = 1630.0
npp0 = 65.0

t_final = 7.5e4

data = []
Ca0s = [Ca0]
Ca05s = [0, Ca05]
ECSs = [3.0, 6.0, 12.0]

for half_idx, C_half in enumerate(Ca05s):
    for atm_idx, Ca in enumerate(Ca0s):
        ccs = [
            ClimateCarbonSystem(
                Alk=Alk,
                Ca05=C_half,
                burnt_fossil_carbon=CF,
                Ca0=Ca * 2.12,
                CL0=CL0,
                npp0=npp0,
                L=3.7 / ECS,
                t_final=t_final,
            )
            for ECS in ECSs
        ]

        for c, ECS in zip(ccs, ECSs):
            c.integrate(c.pi_equilibrium)
            data.append(
                pd.DataFrame(
                    {
                        "Ca": c.Ca / 2.12,
                        "CL": c.CL,
                        "Time": c.t,
                        "Chalf": C_half / 2.12,
                        "Ca0": Ca,
                        "ECS": ECS,
                    }
                )
            )
df = pd.concat(data)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
ax1.plot(
    df[(df.Chalf == 0) & (df.ECS == 3.0)].Time,
    df[(df.Chalf == 0) & (df.ECS == 3.0)].Ca,
    label="ECS = 3K",
    linewidth=3,
)
ax1.plot(
    df[(df.Chalf == 0) & (df.ECS == 6.0)].Time,
    df[(df.Chalf == 0) & (df.ECS == 6.0)].Ca,
    label="ECS = 6K",
    linewidth=3,
)
ax1.plot(
    df[(df.Chalf == 0) & (df.ECS == 12.0)].Time,
    df[(df.Chalf == 0) & (df.ECS == 12.0)].Ca,
    label="ECS = 12K",
    linewidth=3,
)

ax1.axhline(Ca0, color="black", linestyle="--", label="Equilibrium")

ax3.plot(
    df[(df.Chalf == 344.0) & (df.ECS == 3.0)].Time,
    df[(df.Chalf == 344.0) & (df.ECS == 3.0)].Ca,
    label="ECS = 3K",
    linewidth=3,
)
ax3.plot(
    df[(df.Chalf == 344.0) & (df.ECS == 6.0)].Time,
    df[(df.Chalf == 344.0) & (df.ECS == 6.0)].Ca,
    label="ECS = 6K",
    linewidth=3,
)
ax3.plot(
    df[(df.Chalf == 344.0) & (df.ECS == 12.0)].Time,
    df[(df.Chalf == 344.0) & (df.ECS == 12.0)].Ca,
    label="ECS = 12K",
    linewidth=3,
)

ax3.axhline(Ca0, color="black", linestyle="--")

ax2.plot(
    df[(df.Chalf == 0) & (df.ECS == 3.0)].CL,
    df[(df.Chalf == 0) & (df.ECS == 3.0)].Ca,
    zorder=9,
    linewidth=3,
)
ax2.plot(
    df[(df.Chalf == 0) & (df.ECS == 6.0)].CL,
    df[(df.Chalf == 0) & (df.ECS == 6.0)].Ca,
    zorder=8,
    linewidth=3,
)
ax2.plot(
    df[(df.Chalf == 0) & (df.ECS == 12.0)].CL,
    df[(df.Chalf == 0) & (df.ECS == 12.0)].Ca,
    linewidth=3,
)

ax4.plot(
    df[(df.Chalf == 344.0) & (df.ECS == 3.0)].CL,
    df[(df.Chalf == 344.0) & (df.ECS == 3.0)].Ca,
    zorder=9,
    linewidth=3,
)
ax4.plot(
    df[(df.Chalf == 344.0) & (df.ECS == 6.0)].CL,
    df[(df.Chalf == 344.0) & (df.ECS == 6.0)].Ca,
    zorder=8,
    linewidth=3,
)
ax4.plot(
    df[(df.Chalf == 344.0) & (df.ECS == 12.0)].CL,
    df[(df.Chalf == 344.0) & (df.ECS == 12.0)].Ca,
    linewidth=3,
)

ax1.set_yscale("log")
ax1.set_ylim(5.0, 10000.0)
ax1.set_xlim(0.0, t_final)
ax3.set_yscale("log")
ax3.set_ylim(5.0, 10000.0)
ax3.set_xlim(0.0, t_final)

ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_ylim(5.0, 10000.0)
ax2.set_xlim(4e1, 2e4)
ax4.set_yscale("log")
ax4.set_xscale("log")
ax4.set_ylim(5.0, 10000.0)
ax4.set_xlim(4e1, 2e4)


fig.supylabel(r"$C_A$ (ppmv)", fontsize=36, x=0.05)
ax3.set_xlabel("Time (years)", fontsize=24)
ax4.set_xlabel(r"$C_L$ (PgC)", fontsize=24)

ax1.spines["bottom"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax1.spines["left"].set_position(("outward", 10))

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_position(("outward", 10))
ax3.spines["bottom"].set_position(("outward", 10))

ax2.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax2.spines["left"].set_position(("outward", 10))

ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_position(("outward", 10))
ax4.spines["bottom"].set_position(("outward", 10))

handles, labels = ax1.get_legend_handles_labels()
ax2.legend(handles, labels, loc="upper right", frameon=False, fontsize=18)

ax1.text(0.0, 5e3, r"$C_{1/2} = 0$ ppmv", fontsize=24)
ax3.text(0.0, 5e3, r"$C_{1/2} = 344$ ppmv", fontsize=24)

for ax in (ax1, ax2, ax3, ax4):
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=18)

plt.savefig("figures/timeseries_and_phase_plane.pdf")
plt.close()
