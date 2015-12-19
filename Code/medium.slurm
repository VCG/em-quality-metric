#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=8000
#SBATCH -t 1-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu

# add additional commands needed for Lmod and module loads here


# add commands for analyses here
cd /n/home05/haehn/Projects/em-quality-metric/Code/
python run_split_cnn.py -r cluster --patchpath patches_medium --epochs 100 --batchsize 10

# end of program
exit 0;
