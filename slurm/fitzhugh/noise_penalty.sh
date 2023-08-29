#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=2 --lr_gamma=0.9995 --model=fitzhugh --kl-anneal-iters=1000 --num-iters=10000 --beta=10"

variable_args=(
    "--noise-penalty=100"
    "--noise-penalty=125"
    "--noise-penalty=150"
    "--noise-penalty=175"
    "--noise-penalty=200"
    "--noise-penalty=225"
    "--noise-penalty=250"
    "--noise-penalty=275"
    "--noise-penalty=300"
    "--noise-penalty=400"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
