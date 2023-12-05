#!/bin/sh
fixed_args="multistablesde/latent_sde.py --model ornstein --dt=0.01 --t1=5.0 --lr_gamma=0.995 --kl-anneal-iters=1000 --latent-size=1 --num-iters 10000"

variable_args=(
	"--beta=0.01"
	"--beta=0.1"
	"--beta=1"
	"--beta=10"
	"--beta=17.7"
	"--beta=31.6"
	"--beta=56.2"
	"--beta=100"
	"--beta=177.8"
	"--beta=316.2"
	"--beta=562.3"
	"--beta=1000"
	"--beta=10000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
