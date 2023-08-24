#!/bin/sh
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energy --kl-anneal-iters=1000 --data-noise-level=0.135 --num-iters 10000 --beta 100"

variable_args=(
	"--noise-std=0.001"
	"--noise-std=0.005"
	"--noise-std=0.01"
	"--noise-std=0.025"
	"--noise-std=0.05"
	"--noise-std=0.1"
	"--noise-std=0.5"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
