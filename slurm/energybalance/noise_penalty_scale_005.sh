#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy --kl-anneal-iters=1000 --data-noise-level=0.135 --num-iters 5000 --beta=10 --noise-std=0.05"

variable_args=(
    "--noise-penalty=1"
    "--noise-penalty=10"
    "--noise-penalty=100"
    "--noise-penalty=1000"
    "--noise-penalty=10000"
    "--noise-penalty=100000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
