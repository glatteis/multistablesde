#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy --kl-anneal-iters=1000 --data-noise-level=0.135 --num-iters 5000 --beta=10 --num-iters 10000"

variable_args=(
    "--batch-size=2"
    "--batch-size=4"
    "--batch-size=8"
    "--batch-size=16"
    "--batch-size=32"
    "--batch-size=64"
    "--batch-size=128"
    "--batch-size=256"
    "--batch-size=512"
    "--batch-size=1024"
    "--batch-size=2048"
    "--batch-size=4096"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
