#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=2 --lr_gamma=0.9995 --model=fitzhughgamma --kl-anneal-iters=1000 --num-iters=10000 --beta=10"

variable_args=(
    "--noise-penalty=500"
    "--noise-penalty=600"
    "--noise-penalty=700"
    "--noise-penalty=800"
    "--noise-penalty=900"
    "--noise-penalty=1000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
