#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy --kl-anneal-iters=1000 --data-noise-level=0.135 --num-iters 5000 --beta=1000"

variable_args=(
    "--noise-penalty=10000"
    "--noise-penalty=100000"
    "--noise-penalty=1000000"
    "--noise-penalty=10000000"
    "--noise-penalty=100000000"
    "--noise-penalty=1000000000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
