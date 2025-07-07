#!/usr/bin/env python3
"""
Created on Mon Jul  7 12:19:58 2025.

@author: Joe Clarke
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


def single_par_plot(
    data,
    ax,
    colour,
    lw=1.5,
    linestyles=["solid", "dashed"],
    PO=True,
    colour_PO="darkgreen",
    beta=1,
):
    counter = 0
    for i in range(len(data[:, 0]) - 1):
        if not data[i, 3] == data[i + 1, 3] or not data[i, 4] == data[i + 1, 4]:
            print(0, i, data[i, 3])
            if data[i, 3] <= 2:
                if data[i, 3] == 2:
                    data11 = data[counter : i + 1, 0]
                    data22 = data[counter : i + 1, 1]
                    ax.plot(
                        data11,
                        beta * data22,
                        c=colour,
                        linewidth=lw,
                        linestyle=linestyles[int((data[i, 3] - 1) % 2)],
                    )
                else:
                    ax.plot(
                        data[counter : i + 1, 0],
                        beta * data[counter : i + 1, 1],
                        c=colour,
                        linewidth=lw,
                        linestyle=linestyles[int((data[i, 3] - 1) % 2)],
                    )
                    print(1, i)
            else:
                if PO:
                    ax.plot(
                        data[counter : i + 1, 0],
                        beta * data[counter : i + 1, 1],
                        c=colour_PO,
                        linewidth=lw,
                        linestyle=linestyles[int((data[i, 3] - 1) % 2)],
                    )
                    ax.plot(
                        data[counter : i + 1, 0],
                        beta * data[counter : i + 1, 2],
                        c=colour_PO,
                        linewidth=lw,
                        linestyle=linestyles[int((data[i, 3] - 1) % 2)],
                    )
                    print(2, i)
            counter = i + 1

    if data[-1, 3] <= 2:
        ax.plot(
            data[counter:, 0],
            beta * data[counter:, 1],
            c=colour,
            linewidth=lw,
            linestyle=linestyles[int((data[-1, 3] - 1) % 2)],
        )
    else:
        if PO:
            ax.plot(
                data[counter:, 0],
                beta * data[counter:, 1],
                c=colour_PO,
                linewidth=lw,
                linestyle=linestyles[int((data[-1, 3] - 1) % 2)],
            )
            ax.plot(
                data[counter:, 0],
                beta * data[counter:, 2],
                c=colour_PO,
                linewidth=lw,
                linestyle=linestyles[int((data[-1, 3] - 1) % 2)],
            )


data1 = np.genfromtxt("data/XPPAUTO_data/ECS_Ca_1.dat", dtype=float, delimiter=" ")
data2 = np.genfromtxt("data/XPPAUTO_data/ECS_Ca_2.dat", dtype=float, delimiter=" ")
data1[:, 1] = data1[:, 1] / 2.12
data1[:, 2] = data1[:, 2] / 2.12
data2[:, 1] = data2[:, 1] / 2.12
data2[:, 2] = data2[:, 2] / 2.12

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

single_par_plot(
    data1,
    ax1,
    "tab:blue",
    lw=1.5,
    linestyles=["solid", "dashed"],
    PO=True,
    colour_PO="tab:green",
    beta=1,
)
single_par_plot(
    data2,
    ax2,
    "tab:blue",
    lw=1.5,
    linestyles=["solid", "dashed"],
    PO=True,
    colour_PO="tab:green",
    beta=1,
)
ax1.set_xlim(3.0, 15.0)
ax1.set_ylim(1, 20000)
ax2.set_ylim(1, 20000)
ax1.set_yscale("log")
ax2.set_yscale("log")
fig.supxlabel("ECS (K)")
fig.supylabel("CO$_2$ concentration (ppm)")
sns.despine()
fig.tight_layout()
plt.savefig("figures/bifurcation_diagram.pdf")
plt.close()
