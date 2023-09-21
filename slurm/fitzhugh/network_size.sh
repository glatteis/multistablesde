#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=2 --lr_gamma=0.9995 --model=fitzhugh --kl-anneal-iters=1000 --num-iters=10000 --beta=10"

variable_args=(
	"--hidden-size=16"
	"--hidden-size=32"
	"--hidden-size=64"
	"--hidden-size=128"
	"--hidden-size=256"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
