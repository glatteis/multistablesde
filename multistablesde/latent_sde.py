"""Train a latent SDE on a model in models/

Reproduce the toy example in Section 7.2 of https://arxiv.org/pdf/2001.01328.pdf

To run this file, first run the following to install extra requirements:
pip install fire

To run, execute:
python -m examples.latent_sde_lorenz
"""
import logging
import os
import sys
import math
import json
from typing import Sequence
import random

import time

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import pandas as pd

import torchsde

from models.energy_balance import StochasticEnergyBalance
from models.energy_balance_constant import ConstantStochasticEnergyBalance
from models.fitzhugh_nagumo import FitzHughNagumo
from models.fitzhugh_nagumo_gamma import FitzHughNagumoGamma
from models.fitzhugh_nagumo_keno import FitzHughNagumoKeno
from models.geometric_bm import GeometricBM
from models.ornstein_uhlenbeck import OrnsteinUhlenbeck


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self, data_size, latent_size, context_size, hidden_size, scalar_diffusion=False, use_projector=False
    ):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(
            input_size=data_size, hidden_size=hidden_size, output_size=context_size
        )
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        if scalar_diffusion:
            # scalar_diffusion sets the diffusion to be as simple as possible
            self.g_nets = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(1, 1),
                    )
                    for _ in range(latent_size)
                ]
            )
        else:
            # more flexible noise
            self.g_nets = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(1, hidden_size),
                        nn.Softplus(),
                        nn.Linear(hidden_size, 1),
                        nn.Sigmoid(),
                        nn.Linear(1, 1),
                        nn.Softplus(),
                    )
                    for _ in range(latent_size)
                ]
            )
        if use_projector:
            self.projector = nn.Linear(latent_size, data_size, bias=False)
        else:
            self.projector = None

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler", dt=None):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                (ctx,)
                + tuple(self.f_net.parameters())
                + tuple(self.g_nets.parameters())
                + tuple(self.h_net.parameters())
            )
            zs, log_ratio, noise_penalty = torchsde.sdeint_adjoint(
                self,
                z0,
                ts,
                adjoint_params=adjoint_params,
                dt=dt,
                logqp_noise_penalty=True,
                method=method,
            )
        else:
            zs, log_ratio, noise_penalty = torchsde.sdeint(
                self, z0, ts, dt=dt, logqp_noise_penalty=True, method=method
            )

        if hasattr(self, "projector") and self.projector:
            _xs = self.projector(zs)
        else:
            _xs = zs[:, :, 0:1]
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = dt * xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)

        noise = noise_penalty.sum(dim=0).mean(dim=0)

        return log_pxs, logqp0 + logqp_path, noise

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None, dt=None, project=True):
        eps = torch.randn(
            size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device
        )
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={"drift": "h"}, dt=dt, bm=bm)
        if project:
            if hasattr(self, "projector") and self.projector:
                _xs = self.projector(zs)
            else:
                _xs = zs[:, :, 0:1]
        else:
            _xs = zs
        return _xs

    @torch.no_grad()
    def posterior_plot(self, xs, ts, dt=None, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                (ctx,)
                + tuple(self.f_net.parameters())
                + tuple(self.g_nets.parameters())
                + tuple(self.h_net.parameters())
            )
            zs, log_ratio, noise_penalty = torchsde.sdeint_adjoint(
                self,
                z0,
                ts,
                adjoint_params=adjoint_params,
                dt=dt,
                logqp_noise_penalty=True,
                method=method,
            )
        else:
            zs, log_ratio, noise_penalty = torchsde.sdeint(
                self, z0, ts, dt=dt, logqp_noise_penalty=True, method=method
            )

        if hasattr(self, "projector") and self.projector:
            _xs = self.projector(zs)
        else:
            _xs = zs[:, :, 0:1]
        return _xs, log_ratio, noise_penalty


def steps(t0, t1, dt):
    return math.ceil((t1 - t0) / dt)


def make_dataset(
    model, t0, t1, t1_extrapolated, dt, batch_size, noise_std, train_dir, device
):
    data_path = os.path.join(train_dir, "data.pth")
    data_path_csv = os.path.join(train_dir, "data.csv")

    steps_train = steps(t0, t1, dt)
    steps_extrapolated = steps(t0, t1_extrapolated, dt)

    ts = torch.linspace(t0, t1, steps=steps_train, device=device)
    ts_extrapolated = torch.linspace(
        t0, t1_extrapolated, steps=steps_extrapolated, device=device
    )

    bm = torchsde.BrownianInterval(
        t0=t0,
        t1=t1_extrapolated,
        size=(
            batch_size,
            model.state_space_size,
        ),
        device=device,
        levy_area_approximation="space-time",
        entropy=48123,  # seed (generate the same data if models are the same)
    )
    torch.manual_seed(48123)
    xs_extrapolated, mean, std = model.sample(
        batch_size, ts_extrapolated.to(device), normalize=False, device=device, bm=bm
    )
    torch.seed()
    xs_train = xs_extrapolated[0:steps_train, :, :]

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    torch.save(
        {
            "ts_train": ts,
            "ts_extrapolated": ts_extrapolated,
            "xs": xs_extrapolated,
            "dt": dt,
            "mean": mean,
            "std": std,
        },
        data_path,
    )
    pd.DataFrame(
        torch.transpose(
            torch.cat((ts_extrapolated.unsqueeze(1).unsqueeze(1), xs_extrapolated), 1)[
                :, :, 0
            ],
            0,
            1,
        )
        .cpu()
        .numpy()
    ).to_csv(data_path_csv)

    logging.warning(f"Stored data at: {data_path}")

    return xs_train.to(device), ts.to(device).to(device)


def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=100, dt=None):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[1, 0])
    ax02 = fig.add_subplot(gs[0, 1])
    ax03 = fig.add_subplot(gs[1, 1])
    ax04 = fig.add_subplot(gs[2, 0])

    # Left plot: data.
    z = xs.cpu().numpy()
    ax00.plot(z[:, :, 0])
    ax00.set_title("Data")
    xlim = ax00.get_xlim()
    ylim = ax00.get_ylim()

    # Right plot: samples from learned model.
    prior = (
        latent_sde.sample(batch_size=num_samples, ts=ts, bm=bm_vis, dt=dt).cpu().numpy()
    )
    ax01.plot(prior[:, :, 0])
    ax01.set_title("Prior")
    ax01.set_xlim(xlim)
    ax01.set_ylim(ylim)

    posterior, kl, noise_penalty = latent_sde.posterior_plot(xs, ts, dt=dt)
    ax02.plot(posterior.cpu().numpy()[:, :, 0])
    ax02.set_title("Posterior")
    ax02.set_xlim(xlim)
    ax02.set_ylim(ylim)

    ax03.plot(kl.cpu().numpy()[:, :])
    ax03.set_title("KL")

    ax04.plot(noise_penalty.cpu().numpy()[:, :])
    ax04.set_title("Noise Level")

    plt.savefig(img_path)
    plt.close()


def plot_learning(loss, kl, logpxs, noise, lr, kl_sched, img_path):
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 2)
    lossp = fig.add_subplot(gs[0, 0])
    noisep = fig.add_subplot(gs[0, 1])
    klp = fig.add_subplot(gs[1, 0])
    logpxsp = fig.add_subplot(gs[1, 1])
    lrp = fig.add_subplot(gs[2, 0])
    kl_schedp = fig.add_subplot(gs[2, 1])

    lossp.set_title("Loss")
    lossp.plot(loss)

    klp.set_title("KL")
    klp.set_yscale("symlog")
    klp.plot(kl)

    logpxsp.set_title("Log-Likelihoods")
    logpxsp.set_yscale("symlog")
    logpxsp.plot(logpxs)

    noisep.set_title("Noise")
    noisep.plot(noise)

    lrp.set_title("Learning Rate")
    lrp.set_yscale("symlog")
    lrp.plot(lr)

    kl_schedp.set_title("Beta")
    kl_schedp.plot(kl_sched)

    plt.savefig(img_path)
    plt.close()


def main(
    batch_size=1024,
    latent_size=4,
    context_size=32,
    hidden_size=100,
    lr_init=1e-2,
    t0=0.0,
    t1=4.0,
    lr_gamma=0.997,
    num_iters=5000,
    kl_anneal_iters=1000,
    pause_every=500,
    noise_std=0.01,
    adjoint=False,
    train_dir="./dump/" + str(time.time_ns()),
    method="euler",
    viz_samples=30,
    beta=1.0,
    dt=0.01,
    model="energy",
    data_noise_level=None,
    scalar_diffusion=False,
    noise_penalty=0.0,
    noise_penalty_iters=1000,
    experimental_loss=False,
    use_projector=False,
):
    # Save the set configuration for analysis - these are just the locals at
    # the beginning of the execution
    configuration = locals()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "energy": StochasticEnergyBalance(),
        "energyconstant": ConstantStochasticEnergyBalance(),
        "fitzhugh": FitzHughNagumo(),
        "fitzhughgamma": FitzHughNagumoGamma(),
        "fitzhughkeno": FitzHughNagumoKeno(),
        "geometricbm": GeometricBM(),
        "ornstein": OrnsteinUhlenbeck(),
    }
    model_obj = models[model]
    if (isinstance(model_obj, StochasticEnergyBalance) or isinstance(model_obj, ConstantStochasticEnergyBalance)) and data_noise_level is not None:
        model_obj.noise_var = data_noise_level

    sys.setrecursionlimit(1500)
    xs, ts = make_dataset(
        model=model_obj,
        t0=t0,
        t1=t1,
        t1_extrapolated=t1 * 5.0,
        dt=dt,
        batch_size=batch_size,
        noise_std=noise_std,
        train_dir=train_dir,
        device=device,
    )

    # Save configuration
    with open(f"{train_dir}/config.json", "w", encoding="utf8") as f:
        json.dump(configuration, f, ensure_ascii=False, indent=4)

    torch.save(model_obj, os.path.join(train_dir, "data_model_obj.pth"))

    latent_sde = LatentSDE(
        data_size=1,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        scalar_diffusion=scalar_diffusion,
        use_projector=use_projector,
    ).to(device)
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=lr_gamma
    )
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters, maxval=beta)
    noise_penalty_scheduler = LinearScheduler(
        iters=noise_penalty_iters, maxval=noise_penalty
    )

    # Fix the same Brownian motion for visualization.
    bm_vis = torchsde.BrownianInterval(
        t0=t0,
        t1=t1,
        size=(
            viz_samples,
            latent_size,
        ),
        device=device,
        levy_area_approximation="space-time",
    )

    recorded_loss = []
    recorded_kl = []
    recorded_logpxs = []
    recorded_noise = []
    recorded_lr = []
    recorded_kl_sched = []
    recorded_noise_sched = []

    for global_step in tqdm.tqdm(range(1, num_iters + 1)):
        latent_sde.zero_grad()
        log_pxs, log_ratio, noise = latent_sde(
            xs, ts, noise_std, adjoint, method, dt=dt
        )
        if experimental_loss:
            loss = -log_pxs + noise * (log_ratio - noise_penalty_scheduler.val)
        else:
            loss = (
                -log_pxs
                + log_ratio * kl_scheduler.val
                - noise * noise_penalty_scheduler.val
            )
        loss.backward()
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()
        noise_penalty_scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        recorded_loss.append(float(loss))
        recorded_kl.append(float(log_ratio))
        recorded_logpxs.append(float(log_pxs))
        recorded_noise.append(float(noise))
        recorded_lr.append(float(lr_now))
        recorded_kl_sched.append(float(kl_scheduler.val))
        recorded_noise_sched.append(float(noise_penalty_scheduler.val))

        if (global_step % pause_every == 0 and global_step != 0) or global_step == 1:
            logging.warning(
                f"global_step: {global_step:06d}, lr: {lr_now:.5f}, "
                f"log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}"
                f"noise: {noise:.4f}"
            )
            img_path = os.path.join(train_dir, f"{global_step:06d}_model.pdf")
            vis(
                xs[:, 0:30, :],
                ts,
                latent_sde,
                bm_vis,
                img_path,
                num_samples=viz_samples,
                dt=dt,
            )
            img_path2 = os.path.join(train_dir, f"{global_step:06d}_train.pdf")
            plot_learning(
                recorded_loss,
                recorded_kl,
                recorded_logpxs,
                recorded_noise,
                recorded_lr,
                recorded_kl_sched,
                img_path2,
            )
            model_path = os.path.join(train_dir, f"{global_step:06d}_pytorch_model.pth")
            torch.save(latent_sde, model_path)
    # Save final model
    torch.save(latent_sde, os.path.join(train_dir, "model.pth"))

    # Save recorded losses, KL divergences, etc.
    training_info = {
        "loss": recorded_loss,
        "kl": recorded_kl,
        "logpxs": recorded_logpxs,
        "lr": recorded_lr,
        "kl_sched": recorded_kl_sched,
        "noise_sched": recorded_noise_sched,
        "noise": recorded_noise,
    }
    with open(f"{train_dir}/training_info.json", "w", encoding="utf8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print(" ".join(sys.argv))
    fire.Fire(main)
