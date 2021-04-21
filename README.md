# deepdist
Deep learning prediction of protein distance map via multi-classification and regression

**(1) Download DeepDist system package (short path is recommended)**

```
git clone git@github.com:multicom-toolbox/deepdist.git

cd deepdist
```

**(2) Activate system python3.6 environment (required)**

```
if on lewis: sh installation/activate_python3_in_lewis_server.sh

if on multicom: sh installation/activate_python3_in_multicom_server.sh

if on ubuntu system, please run below command first
	sudo apt-get install csh
	sudo ln -s bash /bin/sh.bash 
	sudo mv /bin/sh.bash /bin/sh
	sudo apt-get install tcsh (if already installed, ignore this)
Note: The system is developed and tested under python3.6. 
```

**(3) Configure deepdist (required)**

<h5>There are two setup script, run one of them is enough. User want to predict with fasta input run setup.py, user want to predict with MSA input run setup_v2.py</h5>

```
python setup.py    # include all the database and tool for generate MSA and features.

python setup_v2.py # only include database and tool for generate features, more lightly.(recommand)

python configure.py

sh installation/set_env.sh
```

**(4) Example for predict the deepdist model**

<h5>Case 1: run individual model</h5>

```
sh predictors/individual/pred_deepdist_plm_cpu.sh
sh predictors/individual/pred_deepdist_plm_lewis_cpu.sh

Output directory: example/*fasta name*/pred_map0/rr/

```

<h5>Case 2: example for run ensemble model with fasta input</h5>

```
sh predictors/ensemble/example/pred_deepdist_v1.sh
sh predictors/ensemble/example/pred_deepdist_v1_lewis.sh

Output directory: example/*fasta name*/pred_map_ensem/rr/
```

<h5>Case 3: example for run ensemble model with MSA input</h5>

```
sh predictors/ensemble/example/pred_deepdist_v2.sh
sh predictors/ensemble/example/pred_deepdist_v2_lewis.sh

Output directory: example/*fasta name*/pred_map_ensem/rr/
```

**(6) run whole deepdist project**

<h5>Case 1: run with fasta input</h5>

```
sh predictors/ensemble/pred_deepdist_v1.sh fasta_file output_folder
example:
sh predictors/ensemble/pred_deepdist_v1.sh example/T1019s1.fasta predictors/results/T1019s1

```

<h5>Case 2: run with MSA input</h5>    

```
sh predictors/ensemble/pred_deepdist_v2_construct.sh fasta_file MSA_file output_folder
sh predictors/ensemble/pred_deepdist_v2_dist.sh fasta_file MSA_file output_folder
example:
sh predictors/ensemble/pred_deepdist_v2_construct.sh example/T1019s1.fasta example/T1019s1.aln predictors/results/T1019s1

```

<h5>Case 3: run with MSA input with different model option</h5>

```
python run_deepdist.py -f fasta_file -a MSA_file -o output_dir -m method
example:
python run_deepdist.py -f ./example/T1019s1.fasta -a ./example/T1019s1.aln -o ./predictors/results/test/ -m mul_class_C
Different options for -m:
1.mul_class_C: Predict 10-bins multi-classification distance map (The CASP14 official format)
2.mul_class_G: Predict 42-bins multi-classification distance map (Can replace the distance map in trRosetta npz file by run 
	python ./lib/npy2trRosetta_npz.py -i npy_file -n npz_file -o new_npz_file)
3.mul_label_R: Predict the real-value distance map and the 25-bins multi-classification distance map at the same time. 
	(The improved version of DeepDist1)

```

Note: For more detailed descriptions of the predicted distance map, please check the paper below. If you have any further questions, please feel free to contact the zggc@umsystem.edu for help.

<h2>References</h2>

1. Wu, T., Guo, Z., Hou, J., & Cheng, J. (2021). DeepDist: real-value inter-residue distance prediction with deep residual convolutional network. BMC bioinformatics, 22(1), 1-17. (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-03960-9)

2. Guo, Z., Wu, T., Liu, J., Hou, J., & Cheng, J. (2021). Improving deep learning-based protein distance prediction in CASP14. bioRxiv. (https://www.biorxiv.org/content/10.1101/2021.02.02.429462v1.full)
