import logging
import os
import sys
from typing import Sequence

import fire
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import scipy.stats
import json

import torchsde

# Wildcard import so that imported files find all classes
from latent_sde import *

def draw_marginals(xs_sde, xs_data, file, title):
    bins = np.linspace(-4, 4, 100)
    plt.hist(
        torch.flatten(xs_sde).numpy(),
        bins=bins,
        alpha=0.5,
        label="Latent SDE",
        edgecolor="black",
        linewidth=1.2,
    )
    plt.hist(
        torch.flatten(xs_data).numpy(),
        bins=bins,
        alpha=0.5,
        label="Data",
        edgecolor="black",
        linewidth=1.2,
    )
    plt.legend()
    plt.title(f"Marginals, {title}")
    plt.savefig(file + ".pdf")
    plt.savefig(file + ".pgf")
    plt.close()


def distance_between_histograms(xs_sde, xs_data):
    values_sde = torch.flatten(xs_sde).numpy()
    values_data = torch.flatten(xs_data).numpy()
    return scipy.stats.wasserstein_distance(values_sde, values_data)


def mean(xs):
    return torch.mean(xs, dim=(1, 2))


def std(xs):
    return torch.std(xs, dim=(1, 2))


def draw_mean_var(ts, xs_sde, xs_data, file, title):
    mean_sde = mean(xs_sde)
    conf_sde = std(xs_sde) * 1.96

    mean_data = mean(xs_data)
    conf_data = std(xs_data) * 1.96

    fig, ax = plt.subplots()
    ax.plot(ts, mean_sde, label="Latent SDE", color="green")
    ax.fill_between(
        ts, (mean_sde - conf_sde), (mean_sde + conf_sde), color="green", alpha=0.1
    )

    ax.plot(ts, mean_data, label="Data", color="orange")
    ax.fill_between(
        ts, (mean_data - conf_data), (mean_data + conf_data), color="orange", alpha=0.1
    )

    ax.legend()
    plt.title(f"95% confidence, {title}")

    plt.savefig(file + ".pdf")
    plt.savefig(file + ".pgf")
    plt.close()


def draw_posterior_around_data(ts, xs_posterior, xs_datapoint, file, title):
    fig, ax = plt.subplots()
    mean_posterior = mean(xs_posterior)
    conf_posterior = std(xs_posterior) * 1.96

    ax.plot(ts, mean_posterior, label="Posterior", color="green")
    ax.fill_between(
        ts,
        (mean_posterior - conf_posterior),
        (mean_posterior + conf_posterior),
        color="green",
        alpha=0.1,
    )
    ax.plot(ts, xs_datapoint[:, 0, 0], label="Data", color="orange", linewidth=2.0)

    ax.legend()
    plt.title(f"Posterior around data, {title}")
    plt.savefig(file + ".pdf")
    plt.savefig(file + ".pgf")
    plt.close()
    

def tipping_rate(ts, xs):
    assert xs.size(dim=2) == 1
    mean_xs = mean(xs)
    
    tips_counted = torch.zeros_like(ts)
    
    for time in range(xs.size(dim=0) - 1):
        tips_counted_here = 0
        for batch in range(xs.size(dim=1)):
            mean_at_time = mean_xs[time]
            before = xs[time, batch, 0]
            after = xs[time + 1, batch, 0]
            if (before > mean_at_time and after <= mean_at_time) or (before < mean_at_time and after >= mean_at_time):
                tips_counted_here += 1
        tips_counted[time] = tips_counted_here
    
    return tips_counted

def draw_tipping(ts, xs_data, xs_sde, window_size, file, title):
    tipping_data = tipping_rate(ts, xs_data).unfold(0, window_size, window_size).sum(dim=1)
    tipping_sde = tipping_rate(ts, xs_sde).unfold(0, window_size, window_size).sum(dim=1)

    plt.plot(ts[::window_size], tipping_sde, color="green", label="Latent SDE")
    plt.plot(ts[::window_size], tipping_data, color="orange", label="Data")
    plt.legend()
    plt.title(f"Observed tips, {title}")
    plt.savefig(file + ".pdf")
    plt.savefig(file + ".pgf")
    plt.close()


def main(model=None, data=None, out=None, folder=None):
    # automatically walk through folder and find data.pth / model.pth pairs
    # also run analysis on entire benchmark
    if out == None:
        out = model.replace(".pth", "")
        os.makedirs(out, exist_ok=True)
        
    matplotlib.use('pgf')
    # clear => use default fonts in pgf exports
    matplotlib.rcParams["font.monospace"].clear().clear()
    matplotlib.rcParams["font.serif"].clear()
    matplotlib.rcParams["font.sans-serif"].clear()

    latent_sde = torch.load(model, map_location=torch.device("cpu"))
    tsxs_data = torch.load(data, map_location=torch.device("cpu"))

    ts_train = tsxs_data["ts_train"]
    ts_extrapolated = tsxs_data["ts_extrapolated"]

    dt = tsxs_data["dt"]

    batch_size = tsxs_data["xs"].size(dim=1)
    xs_sde_extrapolated = latent_sde.sample(batch_size, ts_extrapolated, dt=dt)
    xs_data_extrapolated = tsxs_data["xs"]

    datapoint_extrapolated = xs_data_extrapolated[:, 1:2, :]
    datapoint_extrapolated_repeated = datapoint_extrapolated.repeat(1, batch_size, 1)
    posterior_extrapolated, _ = latent_sde.posterior_plot(
        datapoint_extrapolated_repeated, ts_extrapolated
    )

    # assumptions: ts_train[0] == 0, ts_train is evenly spaced
    assert ts_train[0] == 0.0

    intervals = {
        "0_halftrain": (len(ts_train) // 2, "1/2 training timespan"),
        "1_train": (len(ts_train), "training timespan"),
        "2_doubletrain": (len(ts_train) * 2, "2 * training timespan"),
        "3_fivetrain": (len(ts_train) * 5, "5 * training timespan"),
    }

    info = {}

    for name, (interval, title) in intervals.items():
        info_local = {}
        xs_data = tsxs_data["xs"][0:interval, :, :]
        xs_sde = xs_sde_extrapolated[0:interval, :, :]
        ts = ts_extrapolated[0:interval]
        posterior = posterior_extrapolated[0:interval, :, :]
        datapoint = datapoint_extrapolated[0:interval, :, :]

        draw_marginals(xs_data, xs_sde, f"{out}/marginals_{name}", title)
        info_local["wasserstein_distance"] = distance_between_histograms(
            xs_sde, xs_data
        )

        draw_mean_var(ts, xs_sde, xs_data, f"{out}/mean_var_{name}", title)

        draw_posterior_around_data(
            ts, posterior, datapoint, f"{out}/posterior_{name}", title
        )

        draw_tipping(ts, xs_data, xs_sde, 10, f"{out}/tipping_{name}", title)
        
        info_local["tipping_rate_data"] = float(tipping_rate(ts, xs_data).sum())
        info_local["tipping_rate_sde"] = float(tipping_rate(ts, xs_sde).sum())

        info[name] = info_local

    # compute wasserstein distance for entire timeseries
    wasserstein_distances = []
    for interval in range(1, len(ts_extrapolated)):
        xs_data = tsxs_data["xs"][interval : interval + 1, :, :]
        xs_sde = xs_sde_extrapolated[interval : interval + 1, :, :]
        wasserstein_distances.append(distance_between_histograms(xs_sde, xs_data))
    plt.plot(ts_extrapolated[1:], wasserstein_distances)
    plt.title("Wasserstein Distances")
    plt.savefig(f"{out}/wasserstein.pdf")
    plt.savefig(f"{out}/wasserstein.pgf")
    plt.close()

    with open(f"{out}/info.json", "w", encoding="utf8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print(" ".join(sys.argv))
    fire.Fire(main)
