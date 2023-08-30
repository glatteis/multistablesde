#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=2 --lr_gamma=0.9995 --model=fitzhugh --kl-anneal-iters=1000 --num-iters=10000 --use-projector"

variable_args=(
	"--beta=0.01"
	"--beta=0.1"
	"--beta=1"
	"--beta=10"
	"--beta=100"
	"--beta=1000"
	"--beta=10000"
	"--beta=17.7"
	"--beta=31.6"
	"--beta=56.2"
	"--beta=177.8"
	"--beta=316.2"
	"--beta=562.3"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
