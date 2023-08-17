#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy"

variable_args=(
	"--beta=1"
	"--beta=10"
	"--beta=100"
	"--beta=1000"
	"--beta=10000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1

