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
	sudo apt-get install tcsh (if already installed, ignore this)
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

sh predictors/ensemble/example/pred_deepdist1.sh
sh predictors/ensemble/example/pred_deepdist1_lewis.sh

Output directory: example/*fasta name*/pred_map_ensem/rr/
```

<h5>Case 3: example for run ensemble model with MSA input</h5>

```

sh predictors/ensemble/example/pred_deepdist2.sh
sh predictors/ensemble/example/pred_deepdist2_lewis.sh

Output directory: example/*fasta name*/pred_map_ensem/rr/
```

**(6) run whole deepdist project**

<h5>Case 1: run with fasta input</h5>

```
sh predictors/ensemble/pred_deepdist1.sh fasta_file output_folder
example:
sh predictors/ensemble/pred_deepdist1.sh example/T1019s1.fasta predictors/results/T1019s1

```

<h5>Case 2: run with MSA input</h5>

```
sh predictors/ensemble/pred_deepdist2.sh fasta_file MSA_file output_folder
example:
sh predictors/ensemble/pred_deepdist2.sh example/T1019s1.fasta example/T1019s1.aln predictors/results/T1019s1

```