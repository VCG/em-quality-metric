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
#SBATCH -t 7-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/home05/haehn/slurm/out.txt
#SBATCH -e /n/home05/haehn/slurm/err.txt

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
cd /n/home05/haehn/Projects/em-quality-metric/Code/
python run_split_cnn.py -r cluster --patchpath patches_large2new --epochs 500 --batchsize 5000 --learning_rate 0.01 --momentum 0.9 --filters1 64 --filters2 64 --filters3 64 --filtersize1 9 --filtersize2 9 --filtersize3 9 --thirdconvlayer True

# end of program
exit 0;
