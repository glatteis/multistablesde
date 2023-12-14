#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energyconstant --kl-anneal-iters=1000 --num-iters 10000 --beta 10"

variable_args=(
	"--data-noise-level=00"
	"--data-noise-level=20"
	"--data-noise-level=40"
	"--data-noise-level=60"
	"--data-noise-level=80"
	"--data-noise-level=100"
	"--data-noise-level=120"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
