#!/bin/sh
fixed_args="multistablesde/latent_sde.py --model ornstein --dt=0.01 --t1=5.0 --lr_gamma=0.999 --kl-anneal-iters=200 --latent-size=1 --num-iters 5000 --beta 10"

variable_args=(
    "--noise-penalty=0"
    "--noise-penalty=50"
    "--noise-penalty=100"
    "--noise-penalty=150"
    "--noise-penalty=200"
    "--noise-penalty=250"
    "--noise-penalty=300"
    "--noise-penalty=350"
    "--noise-penalty=400"
    "--noise-penalty=450"
    "--noise-penalty=500"
    "--noise-penalty=550"
    "--noise-penalty=600"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
