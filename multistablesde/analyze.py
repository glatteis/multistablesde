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

"""
The analysis script, WIP
"""
def main(model = None, data = None):
    latent_sde = torch.load(model)
    tsxs = torch.load(data)
    print(latent_sde)
    print(tsxs)
    print(latent_sde.sample(10, tsxs["ts"]))

if __name__ == "__main__":
    print(" ".join(sys.argv))
    fire.Fire(main)
