#!/usr/bin/env python3
"""
Created on Wed Oct 15 15:37:51 2025.

@author: Joe Clarke
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, ScalarFormatter

from conceptual_model_with_fossil_carbon import ClimateCarbonSystem


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(10, 8))
    for ECS in [10.2, 10.1, 10.0]:
        ccs = ClimateCarbonSystem(
            allow_T_dependence=True,
            L=3.7 / ECS,
            burnt_fossil_carbon=20 * 2.12,
            t_final=500_000,
            use_T1=False,
        )
        ccs.integrate()

        ax.plot(ccs.t, ccs.Ca, label=f"ECS = {ECS}K")
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 0.5), loc="center left")

    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_title("With Temperature Dependent CO$_2$ solubility")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Atmospheric CO$_2$ (PgC)")
    plt.tight_layout()
    plt.savefig("figures/with_T_dep_solubility.pdf")
    plt.close()
