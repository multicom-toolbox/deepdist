# deepdist
Deep learning prediction of protein distance map via multi-classification and regression

**(1) Download DeepDist system package (short path is recommended)**

```
git clone git@github.com:multicom-toolbox/deepdist.git

cd deepdist
```

**(2) Activate system python3 environment (required)**

```
if on lewis: sh installation/activate_python3_in_lewis_server.sh

if on multicom: sh installation/activate_python3_in_multicom_server.sh

if on ubuntu system, please run below command first
	sudo apt-get install csh
	sudo ln -s bash /bin/sh.bash 
	sudo mv /bin/sh.bash /bin/sh
```

**(3) Configure deepdist (required)**

```
python setup.py

python configure.py

sh installation/set_env.sh
```

**(4) Predict the deepdist model (single target, input fasta (should user change the fasta in shell file), ouput rr file**

<h5>Case 1: run individual model</h5>

```

sh predictors/individual/pred_deepdist_plm_cpu.sh
sh predictors/individual/pred_deepdist_plm_lewis_cpu.sh

Output directory: example/*fasta name*/pred_map0/rr/

```

<h5>Case 2: run ensemble model</h5>

```

sh predictors/ensemble/pred_deepdist_v3rc.sh
sh predictors/ensemble/pred_deepdist_v3rc_lewis.sh

Output directory: example/*fasta name*/pred_map_ensem/rr/

```

**(5) Predict and Evaluate the deepdist model on CASP13 43 FM domain**

<h5>Case 1: run ensemble model</h5>

```

sh predictors/ensemble/evalu_deepdist_v3rc.sh
sh predictors/ensemble/evalu_deepdist_v3rc_lewis.sh

Output directory: predictors/results/ENSEMBLE/

```

**(6) run whole deepdist project**

```
sh predictors/ensemble/pred_dist.sh fasta_file output_folder
example:
sh predictors/ensemble/pred_dist.sh example/T1019s1.fasta predictors/results/T1019s1

```