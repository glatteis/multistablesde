#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energyconstant --kl-anneal-iters=1000 --num-iters 10000 --beta 10"

variable_args=(
	"--data-noise-level=0.0"
	"--data-noise-level=0.1"
	"--data-noise-level=0.2"
	"--data-noise-level=0.3"
	"--data-noise-level=0.4"
	"--data-noise-level=0.5"
	"--data-noise-level=0.6"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
