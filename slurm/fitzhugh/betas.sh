#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=2 --lr_gamma=0.9995 --model=fitzhughkeno --kl-anneal-iters=1000 --num-iters=10000"

variable_args=(
	"--beta=0.01"
	"--beta=0.1"
	"--beta=1"
	"--beta=10"
	"--beta=100"
	"--beta=1000"
	"--beta=10000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
