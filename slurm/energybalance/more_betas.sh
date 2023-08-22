#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy --kl-anneal-iters=1000 --data-noise-level=0.135 --num-iters 5000"

variable_args=(
	"--beta=10"
	"--beta=17.7"
	"--beta=31.6"
	"--beta=56.2"
	"--beta=100"
	"--beta=177.8"
	"--beta=316.2"
	"--beta=562.3"
	"--beta=1000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
