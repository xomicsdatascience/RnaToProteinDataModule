#!/bin/bash

echo "5x2 Run $SLURM_ARRAY_TASK_ID $1"
eval "$(conda shell.bash hook)"
conda activate ax2
cd /home/cranneyc/dataModuleMethods/RnaToProteinDataModule
srun python 5x2_run.py $SLURM_ARRAY_TASK_ID $1
