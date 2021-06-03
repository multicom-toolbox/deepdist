# DeepDist
Deep learning prediction of protein distance map via multi-classification and regression

**(1) Download DeepDist package (a short path for the package is recommended)**

```
git clone git@github.com:multicom-toolbox/deepdist.git

cd deepdist
```

**(2) Install and activate Python 3.6.x environment on Linux (required)**

The installation of Python 3.6.x may be different for different Linux systems. 

**Note**: The system is developed and tested under Python 3.6.x. 

**(3) Make sure bash is installed on Linux**

For most Linux distributions such as Red Hat and CentOS, bash is installed by default. No action is needed. However, because bash is often 
not available on Ubuntu, you can run the following commands to install it:

```
sudo apt-get install csh
sudo ln -s bash /bin/sh.bash 
sudo mv /bin/sh.bash /bin/sh
sudo apt-get install tcsh (if already installed, ignore this)	
```

**(4) Configure DeepDist (required)**

Use `setup_msa.py` to configure DeepDist to take a multiple sequence alignment (MSA) as input to predict distance map. `setup_msa.py` requires downloading a package of about 110 GB
Use `setup_fasta.py` to configure DeepDist to take a fasta sequence as input to predict distance map. `setup_fasta.py` requires downloading a package of about 580 GB

```
Step 1:
python setup_msa.py (recommend) or python setup_fasta.py

Step 2:
python configure.py

Step 3: 
sh ./installation/set_env.sh
```

**(4) Examples for predicting distance maps using DeepDist**

<h4>Case 1: Use an ensemble of multiple individual deep learning models to predict distance map from a MSA using different commands</h4>

Command 1 for classifying distances into 10 bins
```
sh ./predictors/ensemble/pred_deepdist_msa_construct.sh [fasta_file] [MSA_file] [output_folder]

Examples:

(1) sh ./predictors/ensemble/pred_deepdist_msa_construct.sh ./example/T1019s1.fasta ./example/T1019s1.aln ./predictors/results/T1019s1

(2) On a standard Linux system
sh predictors/ensemble/example/pred_deepdist_msa.sh

(3) On Mizzou's Lewis Cluster
sh predictors/ensemble/example/pred_deepdist_msa_lewis.sh
```
Output directory: `example/*fasta name*/pred_map_ensem/`. 

The multi_classification distance file `.npy`, the real-value distance file `.txt`, binary contact file at 8 Angstrom threshold `.txt`, binary contact file `.rr` can be visualized by [ConEVA](https://github.com/multicom-toolbox/ConEVA). 

Command 2 for classifying distances into 42 bins
```
sh ./predictors/ensemble/pred_deepdist_msa_dist.sh [fasta_file] [MSA_file] [output_folder]
```
Command 3 for choosing prediction methods
```
python run_deepdist.py -f [fasta_file] -a [MSA_file] -o [output_dir] -m [method]
```
Different options for -m:

1. `mul_class_C`: Predict 10-bin multi-classification distance map (The CASP14 official format)

2. `mul_class_G`: Predict 42-bin multi-classification distance map (It can be converted to trRosetta format. You can replace the distance map in trRosetta npz file by running `python ./lib/npy2trRosetta_npz.py -i [npy_file] -n [npz_file] -o [new_npz_file]`)

3. `mul_label_R`: Predict the real-value distance map and the 25-bins multi-classification distance map at the same time. 
	(This is the improved version of DeepDist1)

An example:
```
python run_deepdist.py -f ./example/T1019s1.fasta -a ./example/T1019s1.aln -o ./predictors/results/test/ -m mul_class_C
```

<h4>Case 2: Use an ensemble of multiple individual deep learning models to predict distance map from a single sequence in the FASTA format</h4>

```
Command: sh ./predictors/ensemble/pred_deepdist_fasta.sh [fasta_file] [output_folder]

Examples of executing the command included in the pacakge: 
On a standard Linux
sh predictors/ensemble/example/pred_deepdist_fasta.sh

On Mizzou's Lewis Cluster:
sh predictors/ensemble/example/pred_deepdist_fasta_lewis.sh
```
The output is stored in the output directory: `example/*fasta name*/pred_map_ensem/`

<h4>Case 3: Use the deep learning model based on pseudo maximum likelihood (plm) feature set to make prediction. </h4>

```
On a standard Linux
sh predictors/individual/pred_deepdist_plm_cpu.sh # See this script for detailed parameters

On Mizzou's Lewis Cluster
sh predictors/individual/pred_deepdist_plm_lewis_cpu.sh
```
The output is stored in the output directory: `example/*fasta name*/pred_map0/`

Note: The accuracy of the ensemble of multiple deep learning models is generally higher than that of an individual model. 

**(5) Three different ways to generate MSA input for DeepDist**

1.Use DeepMSA to generate a MSA for a protein.
Download and install [DeepMSA](https://zhanglab.dcmb.med.umich.edu/DeepMSA/). This package requires installing large protein sequence databases. 

2.Use [HHblits](https://github.com/soedinglab/hh-suite) to search against a standard protein sequence database created by HHsuite (e.g. UniRef30) to generate MSAs.
The UniRef database created by HHsuite is much smaller than the databases used by DeepMSA. So this approach is faster than DeepMSA, but may be less senstive for some proteins. You can download a recent UniRef database (UniRef30_2020_06_hhsuite.tar.gz) [here](http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/) as follows:
```
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
```
Below is an example of generating a MSA from the database.
```
sh ./scripts/hhblits.sh T1049  /DeepDist_full_path/example/T1049.fasta /DeepDist_full_path/predictors/results/T1049 /full_path/UniRef30_2020_06_hhsuite_database
```
3.Use HHblits to search against the Big Fantastic Database ([BFD](https://bfd.mmseqs.com/)).
The BFD database is very large. Searching a protein against BFD is slow, but more sensitive. 
Below is an example of generating a MSA from BFD:
```
sh ./scripts/hhblits.sh T1049  /DeepDist_full_path/example/T1049.fasta /DeepDist_full_path/predictors/results/T1049 /BFD_database_full_path
```

**(6) Convert a multi-classification distance map into a real-value distance map**

```
python ./lib/mulclass2realdist.py -i [input_mulclass_prediction] -o [output_folder]

Example:
python ./lib/mulclass2realdist.py -i ./example/CASP13_results/mul_class/T0949.npy -o ./predictors/results/1/
```

**(7) Train the network from scratch**

```
python ./lib/train_deepdist_tune_net.py [model_name] [dataset_name] [feature_file_name] [loss_function] [filter_number] [layers] [kernel_size] [outter_epoch] [inner_epoch] [feature_dir] [output_dir] [accuracy_log_dir] [weights] [index

Example:
python ./lib/train_deepdist_tune_net.py 'DEEPDIST_RESRC' 'DEEPDIST' 'feature_to_use_plm_v3' 'categorical_crossentropy_C' 64 20 3 70 1 [feature_dir] ./models/custom/test ./models/custom/ 1 1
```
The feature dir must have following sub-directories:
- `cov`: The folder of covariance feature, suffix `.cov` 
- `plm`:The folder of psudo maximum liklihood feature, suffix `.plm`
- `pre`:The folder of precision feature, suffix `.pre`
- `other`: The folder of DNCON2 features, suffix `.txt`
- `bin_class`: The folder of contact map file, suffix `.txt`
- `mul_class_2_22`: The folder of type 'G' multiclass distance map file, suffix `.npy`

Use the command below to generate distance label from pdb file
```
python ./scripts/generate_label_from_pdb.py -f [fasta_file] -p [pdb_file] -o [output folder] -t [type of multiclass distance map] 

Example:
python ./scripts/generate_label_from_pdb.py -f ./example//T0949.fasta -p ./example//T0949.pdb -o ./predictors/results/label_test/ -t G
```

Note: If you have any further questions, feel free to post it in this repository or contact Zhiye Guo at zggc9@umsystem.edu for help.

<h2>References</h2>

1. Guo, Z., Wu, T., Liu, J., Hou, J., & Cheng, J. (2021). Improving deep learning-based protein distance prediction in CASP14. Bioinformatics. (https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btab355/6271413?guestAccessKey=39f28192-590e-4276-8011-54dc6ffed643)
2. Wu, T., Guo, Z., Hou, J., & Cheng, J. (2021). DeepDist: real-value inter-residue distance prediction with deep residual convolutional network. BMC bioinformatics, 22(1), 1-17.(https://link.springer.com/article/10.1186/s12859-021-03960-9)
