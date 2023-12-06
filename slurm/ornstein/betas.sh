#!/bin/sh
fixed_args="multistablesde/latent_sde.py --model ornstein --dt=0.01 --t1=5.0 --lr_gamma=0.999 --kl-anneal-iters=200 --latent-size=1 --num-iters 5000"

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
