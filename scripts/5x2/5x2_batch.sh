#!/bin/bash

# Define variables
job_name=$1
memory="40G"
num_cores=1
time="2-00:00"

# Submit the job using sbatch with the desired parameters
sbatch \
    --job-name="5x2_$job_name" \
    --mem="$memory" \
    -c "$num_cores" \
    -t "$time" \
    --output="/home/cranneyc/dataModuleMethods/slurmOutputs_5x2/slurm.%x.%j.out" \
    --error="/home/cranneyc/dataModuleMethods/slurmOutputs_5x2/slurm.%x.%j.err" \
    --mail-type=ALL \
    --mail-user=caleb.cranney@cshs.org \
    --array=0-999 \
    _5x2_main_code_run.sh $job_name

