#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=12 
#SBATCH --job-name=ml_proj
#SBATCH --account=cs_gy_9223-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --requeue

module purge

singularity exec --nv --overlay /scratch/vh2308/my_env.ext3:rw \
/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "source /ext3/env.sh; python main.py | tee output_4_mass.txt"