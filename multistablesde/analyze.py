import logging
import os
import sys
from typing import Sequence

import fire
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import scipy.stats
import json
import glob
from pathlib import Path

import torchsde

# Wildcard import so that imported files find all classes
from latent_sde import *

interval_names = {
    "0_firsthalftrain": "$(0, 0.5{t_{train}})$",
    "1_secondhalftrain": "$(0.5{t_{train}}, t_{train})$",
    "2_train": "$(0, t_{train})$",
    "3_doubletrain": "$(0, 2 t_{train})$",
    "4_fivetrain": "$(0, 5 t_{train})$",
    "5_extrapolation": "$(t_{train}, 5 t_{train})$",
}


def draw_marginals(xs_sde, xs_data, file, title):
    bins = np.linspace(
        min(xs_sde.min(), xs_data.min()), max(xs_sde.max(), xs_data.max()), 100
    )
    plt.hist(
        torch.flatten(xs_sde).numpy(),
        bins=bins,
        alpha=0.25,
        label="Latent SDE",
        edgecolor="black",
        color="green",
        linewidth=1.2,
    )
    plt.hist(
        torch.flatten(xs_data).numpy(),
        bins=bins,
        alpha=0.25,
        label="Data",
        edgecolor="black",
        color="orange",
        linewidth=1.2,
    )
    plt.legend()
    # plt.title(f"Marginals, {title}")
    plt.xlabel("Value $u(t)$")
    plt.ylabel("Frequency")
    plt.tight_layout(pad=0.3)
    plt.savefig(file + extension)
    plt.close()


def distance_between_histograms(xs_sde, xs_data):
    values_sde = torch.flatten(xs_sde).numpy()
    values_data = torch.flatten(xs_data).numpy()
    return scipy.stats.wasserstein_distance(values_sde, values_data)


def mean(xs):
    return torch.mean(xs, dim=(1, 2))


def std(xs):
    return torch.std(xs, dim=(1, 2))


def bifurcation(xs):
    flattened_xs = xs.flatten()
    search_space = np.linspace(-1.0, 1.0, num=100)
    histogram, _ = np.histogram(flattened_xs, bins=search_space)
    min_point = np.argmin(histogram)
    #assert min_point != 0 and min_point != len(search_space) - 1

    return search_space[min_point]


def draw_xs(ts, xs, file, title, save=True):
    plt.plot(ts, xs, label="Data", linewidth=0.5)
    plt.xlabel("Time $t$")
    plt.ylabel("Value $u(t)$")

    plt.ylim(-4, 4)

    # plt.title(title)
    if save:
        plt.tight_layout(pad=0.3)
        plt.savefig(file + extension)
        plt.close()


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
    plt.xlabel("Time $t$")
    plt.ylabel("Value $u(t)$")
    # plt.title(f"95% confidence, {title}")

    plt.tight_layout(pad=0.3)
    plt.savefig(file + extension)
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

    plt.xlabel("Time $t$")
    plt.ylabel("Value $u(t)$")
    # plt.title(f"Posterior around data, {title}")
    plt.tight_layout(pad=0.3)
    plt.savefig(file + extension)
    plt.close()


def tipping_rate(ts, xs):
    assert xs.size(dim=2) == 1

    tips_counted = torch.zeros_like(ts)

    bif = bifurcation(xs)

    xs_relative = (xs - bif)[:, :, 0]
    # now, positive == above bifurcation, negative == below
    cat_zeros = torch.zeros((1, xs_relative.size(dim=1)))
    xs_relative_1 = torch.cat([cat_zeros, xs_relative])
    xs_relative_2 = torch.cat([xs_relative, cat_zeros])
    xs_diff = xs_relative_1 * xs_relative_2
    tips_counted = torch.vmap(lambda x: x < 0)(xs_diff).sum(dim=1)[1:]

    # checked against this simpler, slower implementation :-)
    # for time in range(xs.size(dim=0) - 1):
    #     tips_counted_here = 0
    #     before = xs[time, :, 0]
    #     after = xs[time + 1, :, 0]
    #     for batch in range(xs.size(dim=1)):
    #         before = xs[time, batch, 0]
    #         after = xs[time + 1, batch, 0]
    #         if (before > bif and after <= bif) or (before < bif and after >= bif):
    #             tips_counted_here += 1
    #     tips_counted[time] = tips_counted_here

    # print(tips_counted == tips_counted_alternative)

    dt = ts[1] - ts[0]
    return tips_counted / (dt * (ts[-1] - ts[0]))


def draw_tipping(ts, xs_sde, xs_data, window_size, file, title):
    tipping_data = (
        tipping_rate(ts, xs_data).unfold(0, window_size, window_size).mean(dim=1)
    )
    tipping_sde = (
        tipping_rate(ts, xs_sde).unfold(0, window_size, window_size).mean(dim=1)
    )

    plt.plot(ts[::window_size], tipping_sde, color="green", label="Latent SDE")
    plt.plot(ts[::window_size], tipping_data, color="orange", label="Data")
    plt.legend()
    plt.xlabel("Time $t$")
    plt.ylabel("Tipping rate")
    # plt.title(f"Observed tips, {title}")
    plt.tight_layout(pad=0.3)
    plt.savefig(file + extension)
    plt.close()


def run_individual_analysis(model, data, show_params=False):
    out = model.replace(".pth", "")
    os.makedirs(out, exist_ok=True)
    print(f"Writing individual analysis to folder {out}")

    latent_sde = torch.load(model, map_location=torch.device("cpu"))
    if show_params:
        print(f"Parameters of model {model}:")
        for name, param in latent_sde.named_parameters():
            if param.requires_grad:
                print(name, param)
    tsxs_data = torch.load(data, map_location=torch.device("cpu"))

    ts_train = tsxs_data["ts_train"]
    ts_extrapolated = tsxs_data["ts_extrapolated"]

    dt = tsxs_data["dt"]

    batch_size = tsxs_data["xs"].size(dim=1)
    xs_sde_extrapolated = latent_sde.sample(batch_size, ts_extrapolated, dt=dt)
    xs_data_extrapolated = tsxs_data["xs"]

    datapoint_extrapolated = xs_data_extrapolated[:, 1:2, :]
    datapoint_extrapolated_repeated = datapoint_extrapolated.repeat(1, batch_size, 1)
    posterior_extrapolated, _, _ = latent_sde.posterior_plot(
        datapoint_extrapolated_repeated, ts_extrapolated
    )

    if latent_sde.pz0_mean.shape[1:][0] == 2:
        draw_phase_portrait(latent_sde.h, np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), out)

    # assumptions: ts_train[0] == 0, ts_train is evenly spaced
    assert ts_train[0] == 0.0

    intervals = {
        "0_firsthalftrain": (0, len(ts_train) // 2),
        "1_secondhalftrain": (len(ts_train) // 2, len(ts_train)),
        "2_train": (0, len(ts_train)),
        "3_doubletrain": (0, len(ts_train) * 2),
        "4_fivetrain": (0, len(ts_train) * 5),
        "5_extrapolation": (len(ts_train), len(ts_train) * 5),
    }

    info = {}

    for name, interval in intervals.items():
        title = interval_names[name]

        info_local = {}
        xs_data = tsxs_data["xs"][interval[0] : interval[1], :, :]
        xs_sde = xs_sde_extrapolated[interval[0] : interval[1], :, :]
        ts = ts_extrapolated[interval[0] : interval[1]]
        posterior = posterior_extrapolated[interval[0] : interval[1], :, :]
        datapoint = datapoint_extrapolated[interval[0] : interval[1], :, :]

        draw_marginals(xs_sde, xs_data, f"{out}/marginals_{name}", title)
        info_local["wasserstein_distance"] = distance_between_histograms(
            xs_sde, xs_data
        )

        draw_xs(ts, xs_sde[:, 0:30, 0], f"{out}/prior_{name}", f"Prior, {title}")
        draw_xs(ts, xs_data[:, 0:30, 0], f"{out}/data_{name}", f"Data, {title}")

        draw_mean_var(ts, xs_sde, xs_data, f"{out}/mean_var_{name}", title)

        draw_posterior_around_data(
            ts, posterior, datapoint, f"{out}/posterior_{name}", title
        )

        draw_tipping(ts, xs_sde, xs_data, 5, f"{out}/tipping_{name}", title)

        info_local["tipping_rate_data"] = float(tipping_rate(ts, xs_data).sum())
        info_local["tipping_rate_sde"] = float(tipping_rate(ts, xs_sde).sum())

        info_local["bifurcation_data"] = bifurcation(xs_data)
        info_local["bifurcation_sde"] = bifurcation(xs_sde)

        info[name] = info_local

    # compute wasserstein distance for entire timeseries
    wasserstein_distances = []
    for interval in range(1, len(ts_extrapolated)):
        xs_data = tsxs_data["xs"][interval : interval + 1, :, :]
        xs_sde = xs_sde_extrapolated[interval : interval + 1, :, :]
        wasserstein_distances.append(distance_between_histograms(xs_sde, xs_data))
    plt.plot(ts_extrapolated[1:], wasserstein_distances)
    plt.xlabel("Time $t$")
    plt.xlabel("Wasserstein Distance")
    # plt.title("Wasserstein Distances")
    plt.tight_layout(pad=0.3)
    plt.savefig(f"{out}/wasserstein" + extension)
    plt.close()

    with open(f"{out}/info.json", "w", encoding="utf8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def draw_param_to_info_both(
    configs,
    infos,
    ts,
    ts_title,
    param_name,
    param_title,
    info_name,
    info_title,
    out,
    xscale="linear",
):
    params = [x[param_name] for x in configs]
    sorted_params = sorted(params)
    # sort by the param, so first zip...
    tipping_rates_sde_sorted = sorted(
        zip(params, [x[ts][f"{info_name}_sde"] for x in infos])
    )
    tipping_rates_data_sorted = sorted(
        zip(params, [x[ts][f"{info_name}_data"] for x in infos])
    )
    # and then choose second item
    tipping_rates_sde = list(zip(*tipping_rates_sde_sorted))[1]
    tipping_rates_data = list(zip(*tipping_rates_data_sorted))[1]
    plt.plot(sorted_params, tipping_rates_sde, label="Latent SDE", color="green")
    plt.plot(sorted_params, tipping_rates_data, label="Data", color="orange")
    plt.xlabel(param_title)
    plt.xscale(xscale)
    plt.ylabel(info_title)
    plt.legend()

    # plt.title(f"{param_title} to {info_title}, {ts_title}")
    plt.tight_layout(pad=0.3)
    plt.savefig(f"{out}/{info_name}_{param_name}_{ts}" + extension)
    plt.close()


def draw_param_to_info(
    configs,
    infos,
    ts,
    ts_title,
    param_name,
    param_title,
    info_name,
    info_title,
    out,
    xscale="linear",
    save=True,
):
    params = [x[param_name] for x in configs]
    sorted_params = sorted(params)
    # sort by the param, so first zip...
    infos_sorted = sorted(zip(params, [x[ts][info_name] for x in infos]))
    # and then choose second item
    info_values = list(zip(*infos_sorted))[1]
    plt.plot(sorted_params, info_values, label=ts_title)
    plt.xlabel(param_title)
    plt.xscale(xscale)
    plt.ylabel(info_title)
    if save:
        # plt.title(f"{param_title} to {info_title}, {ts_title}")
        plt.tight_layout(pad=0.3)
        plt.savefig(f"{out}/{info_name}_{param_name}_{ts}" + extension)
        plt.close()


def scatter_param_to_training_info(
    configs, training_infos, param_name, param_title, out, xscale="linear"
):
    params = [x[param_name] for x in configs]
    sorted_params = sorted(params)

    training_info_names = {
        "kl": ("KL Divergence", "green"),
        "logpxs": ("Log-Likelihood", "orange"),
        "noise": ("Diffusion Size", "blue"),
    }

    for training_info_name, (
        training_info_title,
        color,
    ) in training_info_names.items():
        if training_info_name not in training_infos[0].keys():
            print(f"No {training_info_name} training info, skippping...")
            continue
        # sort by the param, so first zip...
        infos_sorted = sorted(
            zip(params, [x[training_info_name][-1] for x in training_infos])
        )
        # and then choose second item
        info_values = list(zip(*infos_sorted))[1]
        plt.plot(sorted_params, info_values, color=color)
        # plt.title(f"{param_title} to {training_info_title}")
        plt.xlabel(param_title)
        plt.ylabel(training_info_title)
        plt.xscale(xscale)
        plt.tight_layout(pad=0.3)
        plt.savefig(
            f"{out}/training_info_{param_name}_{training_info_name}" + extension
        )
        plt.close()


def run_summary_analysis(model_folders, out):
    print(f"Writing summary analysis to folder {out}")
    # self-reported kwargs of simulations
    config_jsons = [os.path.join(x, "config.json") for x in model_folders]
    # summary statistics that we just generated
    info_jsons = [os.path.join(x, "model/info.json") for x in model_folders]
    # training info
    training_info_jsons = [os.path.join(x, "training_info.json") for x in model_folders]

    configs = [json.loads(Path(f).read_text()) for f in config_jsons]
    infos = [json.loads(Path(f).read_text()) for f in info_jsons]
    training_infos = [json.loads(Path(f).read_text()) for f in training_info_jsons]

    timespans = infos[0].keys()

    params = {
        "beta": ("$\\beta$", "log"),
        "context_size": ("Context Size", "linear"),
        "data_noise_level": ("Data Noise Level", "linear"),
        "noise_std": ("Noise Standard Deviation", "log"),
        "noise_penalty": ("Noise Penalty", "linear"),
        "batch_size": ("Size of Dataset", "log"),
    }

    for param_name, (param_title, xscale) in params.items():
        if not param_name in configs[0].keys():
            print(f"No {param_name}, skipping")
            continue
        params = [x[param_name] for x in configs]
        if None in params:
            print(f"None in {param_name}, skipping")
            continue
        if params.count(params[0]) == len(params):
            print(f"Only same value in {param_name}, skipping")
            continue

        for ts in timespans:
            draw_param_to_info_both(
                configs,
                infos,
                ts,
                interval_names[ts],
                param_name,
                param_title,
                "tipping_rate",
                "Tipping Rate",
                out,
                xscale=xscale,
            )

            draw_param_to_info_both(
                configs,
                infos,
                ts,
                interval_names[ts],
                param_name,
                param_title,
                "bifurcation",
                "Bifurcation",
                out,
                xscale=xscale,
            )

            draw_param_to_info(
                configs,
                infos,
                ts,
                interval_names[ts],
                param_name,
                param_title,
                "wasserstein_distance",
                "Wasserstein Distance",
                out,
                xscale=xscale,
            )

        # custom summary plots!

        old_figsize = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = (5, 1.8)
        # custom wasserstein distance plot
        for ts in ["1_secondhalftrain", "5_extrapolation"]:
            draw_param_to_info(
                configs,
                infos,
                ts,
                interval_names[ts],
                param_name,
                param_title,
                "wasserstein_distance",
                "Wasserstein Distance",
                out,
                xscale=xscale,
                save=False,
            )
        plt.tight_layout(pad=0.3)
        plt.legend()
        plt.savefig(f"{out}/custom_wasserstein" + extension)
        plt.close()

        # custom tipping rate plot
        for ts in ["1_secondhalftrain", "5_extrapolation"]:
            draw_param_to_info(
                configs,
                infos,
                ts,
                f"Latent SDE, {interval_names[ts]}",
                param_name,
                param_title,
                "tipping_rate_sde",
                "Tipping Rate",
                out,
                xscale=xscale,
                save=False,
            )
            draw_param_to_info(
                configs,
                infos,
                ts,
                f"Data, {interval_names[ts]}",
                param_name,
                param_title,
                "tipping_rate_data",
                "Tipping Rate",
                out,
                xscale=xscale,
                save=False,
            )
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(pad=0.3)

        plt.savefig(f"{out}/custom_tipping" + extension)
        plt.close()

        plt.rcParams["figure.figsize"] = old_figsize

        scatter_param_to_training_info(
            configs, training_infos, param_name, param_title, out, xscale=xscale
        )

def draw_phase_portrait(h, y1, y2, out):
    # adapted from https://kitchingroup.cheme.cmu.edu/blog/2013/02/21/Phase-portraits-of-a-system-of-ODEs/
    g1, g2 = np.meshgrid(y1, y2)
    out1 = np.zeros(g1.shape)
    out2 = np.zeros(g2.shape)
    t = 0
    for i in range(len(y1)):
        for j in range(len(y2)):
            x = g1[i, j]
            y = g2[i, j]
            out1[i, j], out2[i, j] = map(float, h(t, torch.tensor([[x, y]], dtype=torch.float32))[0])

    plt.quiver(g1, g2, out1, out2, (out1**2 + out2**2)**0.5)
    plt.xlabel('$y_1$')
    plt.ylabel('$y_2$')
    plt.tight_layout(pad=0.3)
    plt.savefig(f"{out}/phase_portrait" + extension)

def main(
    model=None, data=None, folder=None, pgf=False, only_summary=False, show_params=False
):
    global extension
    global plt
    if pgf:
        matplotlib.use("pgf")
        extension = ".pgf"
    else:
        extension = ".pdf"
    import matplotlib.pyplot as plt

    if pgf:
        # from https://jwalton.info/Matplotlib-latex-PGF/
        plt.rcParams.update(
            {
                "font.family": "serif",  # use serif/main font for text elements
                "text.usetex": True,  # use inline math for ticks
                "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            }
        )
    plt.rcParams.update(
        {
            "figure.figsize": (2.8, 1.8),
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.top": False,
            "ytick.right": False,
        }
    )

    # automatically walk through folder and find data.pth / model.pth pairs
    # also run analysis on entire benchmark

    models_and_data = []
    model_folders = None

    if folder is None:
        # just run the analysis script on this one file
        models_and_data = [(model, data)]
    else:
        model_files = glob.glob(f"{folder}/**/model.pth", recursive=True)
        model_folders = [os.path.dirname(x) for x in model_files]
        data_files = [os.path.join(x, "data.pth") for x in model_folders]
        assert all([os.path.exists(x) for x in data_files])
        models_and_data = list(zip(model_files, data_files))

    if not only_summary:
        for model, data in models_and_data:
            run_individual_analysis(model, data, show_params=show_params)

    # if we ran a batch analyze, run the meta-analysis as well
    if folder is not None:
        summary_folder = os.path.join(folder, "summary")
        os.makedirs(summary_folder, exist_ok=True)
        run_summary_analysis(model_folders, summary_folder)


if __name__ == "__main__":
    print(" ".join(sys.argv))
    fire.Fire(main)
