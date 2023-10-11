#!/bin/sh
fixed_args="multistablesde/latent_sde.py --model ornstein --dt=0.25 --t1=50.0 --decay=0.995 --kl-anneal-iters=1000 --latent-size=2 --num-iters 10000"

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
