import logging
import os
import sys
from typing import Sequence

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal

import torchsde

from latent_sde import *

def draw_marginals(xs_sde, xs_data, file):
    bins = np.linspace(-4, 4, 100)
    plt.hist(torch.flatten(xs_sde).numpy(), bins=bins, alpha=0.5, label="Latent SDE", edgecolor='black', linewidth=1.2)
    plt.hist(torch.flatten(xs_data).numpy(), bins=bins, alpha=0.5, label="Data", edgecolor='black', linewidth=1.2)
    plt.savefig(file)
    plt.close()

"""
The analysis script, WIP
"""
def main(model = None, data = None, out = None):
    latent_sde = torch.load(model, map_location=torch.device("cpu"))
    tsxs_data = torch.load(data, map_location=torch.device("cpu"))
    
    ts = tsxs_data["ts"]
    
    marginal_size = 1000
    xs_sde = latent_sde.sample(marginal_size, ts)
    xs_data = tsxs_data["xs"]
    
    draw_marginals(xs_sde, xs_data, f"{out}/marginals.pdf")

if __name__ == "__main__":
    print(" ".join(sys.argv))
    fire.Fire(main)
