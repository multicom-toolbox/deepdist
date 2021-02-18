#!/bin/bash -l
if [ $# -ne 3 ]
then
        echo "need two parameters: input fasta file, MSA file, output directory."
        exit 1
fi

export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
##GLOBAL_FLAG
global_dir=/mnt/data/zhiye/Python/DeepDist
## ENV_FLAG
source $global_dir/env/deepdist_virenv/bin/activate
models_dir[0]=$global_dir/models/pretrain/MULTICOM-DIST/1.dres152_deepcov_cov_ccmpred_pearson_pssm/
models_dir[1]=$global_dir/models/pretrain/MULTICOM-DIST/2.dres152_deepcov_plm_pearson_pssm/
models_dir[2]=$global_dir/models/pretrain/MULTICOM-DIST/3.res152_deepcov_pre_freecontact/
models_dir[3]=$global_dir/models/pretrain/MULTICOM-DIST/4.res152_deepcov_other/
## DBTOOL_FLAG
db_tool_dir=/mnt/data/zhiye/Python/DNCON4_db_tools/
printf "$global_dir\n"
fasta=$1
aln_file=$2
output_dir=$3

#################database_path fasta model outputdir method option
python $global_dir/lib/Model_predict_v2.py $db_tool_dir $fasta $aln_file ${models_dir[@]} $output_dir 'mul_class_G' 
