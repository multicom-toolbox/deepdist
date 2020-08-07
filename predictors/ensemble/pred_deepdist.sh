#!/bin/sh
if [ $# -ne 2 ]
then
        echo "need two parameters: input fasta file, output directory."
        exit 1
fi

export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
##GLOBAL_FLAG
deepdist_dir=/data/casp14/DeepDist/
fasta=$1
outdir=$2
## ENV_FLAG
source $deepdist_dir/env/deepdist_virenv/bin/activate 

python $deepdist_dir/lib/run_deepdist.py -f $fasta -o $outdir 