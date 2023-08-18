#!/bin/sh
fixed_args="multistablesde/latent_sde.py --model ornstein --dt=0.05 --t1=10.0 --decay=0.999 --kl-anneal-iters=1000 --latent-size=2"

variable_args=(
	"--beta=0.1"
	"--beta=1"
	"--beta=10"
	"--beta=100"
	"--beta=1000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
