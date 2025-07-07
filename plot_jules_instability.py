#!/usr/bin/env python3

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
import pandas as pd
import seaborn as sns

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

data_dir = "data/runs/"
Ca0 = 286.085
fit_period = slice(25, None)


def ff(E, q1, q2, q3):
    return (q1 + q2 * E) / (1 + q3 * E)


def critical_ecs(popt, noise=0.0):
    q1, q2, q3 = popt
    return -(q1 + noise) / (q2 + q3 * noise)


if __name__ == "__main__":
    jules_files = list(
        filter(
            lambda jf: float(jf.split("_")[-1][:-3]) <= 15.0,
            sorted(os.listdir(data_dir), key=lambda jf: float(jf.split("_")[-1][:-3])),
        )
    )

    ECSs = [float(jf.split("_")[-1][:-3]) for jf in jules_files]

    print(f"Discovered ECSs: {ECSs}")
    colors = sns.color_palette("tab20", len(ECSs))
    print(f"{len(ECSs)} files found, {len(colors)} colors found")

    rates = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for jf, ECS, color in zip(jules_files, ECSs, colors):
        print(f"On ECS = {ECS}")
        data = xr.open_dataset(data_dir + jf)
        dCa = data.co2_ppmv.isel(scalar=0) - Ca0
        fit = scipy.stats.linregress(
            10 * dCa.decade[fit_period], np.log(dCa[fit_period])
        )
        rates.append(fit.slope)

        ax.plot(10 * dCa.decade, dCa, color=color, label=f"ECS = {ECS}K")
        ax.plot(
            10 * dCa.decade,
            np.exp(fit.slope * 10 * dCa.decade) * np.exp(fit.intercept),
            color=color,
            linestyle=":",
        )

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Perturbed Atmospheric Carbon (ppm)")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5), reverse=True)
    plt.tight_layout()
    plt.savefig("figures/jules_log_dCa_fit_many.pdf")
    plt.close()

    df = pd.DataFrame(data={"ECS": ECSs, "rates": rates})
    popt, pcov = scipy.optimize.curve_fit(ff, ECSs, rates)
    sigma = (np.array(rates) - ff(np.array(ECSs), *popt)).std()
    ecsfitting = np.linspace(min(ECSs), max(ECSs))
    pred = ff(ecsfitting, *popt)

    fig, ax = plt.subplots()
    df.plot.scatter(x="ECS", y="rates", ax=ax, color="black")

    ax.plot(ecsfitting, pred, color="blue")

    ax.set_xlabel("ECS (K)")
    ax.set_ylabel("Growth Rate (yr$^{-1}$)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("figures/jules_growth_rate_fit_many.pdf")
    plt.close()

    reps = 1_000_000
    popts = np.random.multivariate_normal(popt, pcov, size=reps)
    noises = np.random.normal(scale=sigma, size=reps)
    crits = np.fromiter(
        (critical_ecs(p, n) for p, n in zip(popts, noises)), dtype="float", count=reps
    )

    plt.hist(crits, np.ceil(np.sqrt(reps)).astype(int), density=True, histtype="step")
    plt.xlabel("Critical ECS (K)")
    plt.ylabel("Density")
    plt.savefig("figures/jules_critical_density.pdf")
    plt.close()

    alpha = 0.05  # 95% CI
    best = crits.mean()
    lower, upper = np.quantile(crits, [alpha, 1 - alpha])
    print(f"Critical ECS = {best}")
    print(f"{(1 - alpha)*100}% confidence interval: [{lower}, {upper}]")
    print(f"Assuming a symmetric range: ±{0.5 * (upper - lower)}")
