#!/bin/bash -l
#SBATCH -J  PRED
#SBATCH -o PRED-%j.out
#SBATCH -p Lewis,hpc4,hpc5
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 2-00:00
#SBATCH --mem 20G

module load cuda/cuda-9.0.176
module load cudnn/cudnn-7.1.4-cuda-9.0.176
export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
##GLOBAL_FALG
global_dir=/mnt/data/zhiye/Python/DeepDist
## ENV_FLAG
source $global_dir/env/dncon4_virenv/bin/activate
models_dir[0]=$global_dir/models/pretrain/dres34_deepcov_plm_temp_test/
output_dir=$global_dir/predictors/results/PLM/
fasta=$global_dir/example/T0771.fasta
## DBTOOL_FLAG
db_tool_dir=/mnt/data/zhiye/Python/DNCON4_db_tools/
printf "$global_dir\n"

#################CV_dir output_dir dataset database_path
python $global_dir/lib/Model_predict.py $db_tool_dir $fasta ${models_dir[@]} $output_dir 'real_dist_limited16'
