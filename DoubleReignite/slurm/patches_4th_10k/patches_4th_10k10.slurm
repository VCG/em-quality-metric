#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=24000
#SBATCH -t 7-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/home05/haehn/slurm/out-b32df392-d9b9-4f66-87a4-1ef38b26a628.txt
#SBATCH -e /n/home05/haehn/slurm/err-b32df392-d9b9-4f66-87a4-1ef38b26a628.txt

# add additional commands needed for Lmod and module loads here
source new-modules.sh
#module load gcc/4.8.2-fasrc01 python/2.7.9-fasrc01
module load Anaconda/2.1.0-fasrc01
#module load cuda/7.5-fasrc01
export CUDA_HOME=/usr/local/cuda-7.0
export CUDA_LIB=/usr/local/cuda-7.0/lib64
export CUDA_INCLUDE=/usr/local/cuda-7.0/include
export PATH=/usr/local/cuda-7.0/bin:$PATH
export CPATH=/usr/local/cuda-7.0/include:$CPATH
export FPATH=/usr/local/cuda-7.0/include:$FPATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-7.0/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LIBRARY_PATH


# add commands for analyses here
cd /n/home05/haehn/Projects/em-quality-metric/Reignite/
python train.py -r cluster --patchpath patches_4th_10k --epochs 500 --batchsize 100 --learning_rate 1e-05 --momentum 0.9 --filters1 32 --filters2 32 --filters3 32 --filtersize1 5 --filtersize2 5 --filtersize3 5 --thirdconvlayer True --uuid b32df392-d9b9-4f66-87a4-1ef38b26a628

# end of program
exit 0;
