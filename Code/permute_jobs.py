#
# LAUNCH THE JOBS WITH:
# for f in *.slurm; do sbatch $f; done
#

import sys
from string import Template
import uuid

PATCH_PATH = 'patches_large_sr2'
OUTPUT_PATH = 'slurm/'+PATCH_PATH+'/'

slurm_header = """#!/bin/bash
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
#SBATCH -o /n/home05/haehn/slurm/out-$uuid.txt
#SBATCH -e /n/home05/haehn/slurm/err-$uuid.txt
#SBATCH -x holyseasgpu[01-03]

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

"""

slurm_body = Template("""
# add commands for analyses here
cd /n/home05/haehn/Projects/em-quality-metric/Code/
python run_split_cnn.py -r cluster --patchpath $patchpath --epochs $epochs --batchsize $batchsize --learning_rate $learning_rate --momentum $momentum --filters1 $no_filters --filters2 $no_filters --filters3 $no_filters --filtersize1 $filter_size --filtersize2 $filter_size --filtersize3 $filter_size --thirdconvlayer $thirdconvlayer --uuid $uuid

# end of program
exit 0;
""")


epochs = [500]
batchsize = [100]#,1000]#,5000]#[100, 500, 1000]
learning_rate = [0.00001, 0.0001]#, 0.01]#[0.000001, 0.00001, 0.0001, 0.001, 0.01]
momentum = [0.9, 0.95]
thirdconvlayer = [True]
no_filters = [16, 32]
filter_size = [9, 13, 17]

no_jobs = 0

for e in epochs:
  for b in batchsize:
    for l in learning_rate:
      for m in momentum:
        for c in thirdconvlayer:
          for n in no_filters:
            for s in filter_size:
              no_jobs += 1
              uniqueid = str(uuid.uuid4())

              new_slurm_body = slurm_body.substitute(patchpath=PATCH_PATH, epochs=e, batchsize=b, learning_rate=l, momentum=m, thirdconvlayer=c, no_filters=n, filter_size=s, uuid=uniqueid)
              slurm = slurm_header.replace('$uuid', uniqueid) + new_slurm_body

              with open(OUTPUT_PATH+PATCH_PATH+str(no_jobs)+'.slurm', 'w') as f:
                f.write(slurm)


print no_jobs, 'slurms generated.'
