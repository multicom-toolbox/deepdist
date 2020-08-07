#!/bin/bash -l
export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
##GLOBAL_FLAG
global_dir=/mnt/data/zhiye/Python/DeepDist
## ENV_FLAG
source $global_dir/env/deepdist_virenv/bin/activate
models_dir[0]=$global_dir/models/pretrain/deepdist_v3rc_GR/1.dres152_deepcov_cov_ccmpred_pearson_pssm/
models_dir[1]=$global_dir/models/pretrain/deepdist_v3rc_GR/2.dres152_deepcov_plm_pearson_pssm/
models_dir[2]=$global_dir/models/pretrain/deepdist_v3rc_GR/3.res152_deepcov_pre_freecontact/
models_dir[3]=$global_dir/models/pretrain/deepdist_v3rc_GR/4.res152_deepcov_other/
output_dir=$global_dir/predictors/results/T1019s1/
fasta=$global_dir/example/T1019s1.fasta
## FEATURE_FLAG
feature_dir=/mnt/data/zhiye/Python/DNCON4_db_tools/
printf "$global_dir\n"

#################database_path fasta model outputdir method option
python $global_dir/lib/Model_predict.py $db_tool_dir $fasta ${models_dir[@]} $output_dir 'mul_lable_R' 'ALN'
