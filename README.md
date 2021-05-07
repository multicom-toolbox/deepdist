# DeepDist
Deep learning prediction of protein distance map via multi-classification and regression

**(1) Download DeepDist package (a short path for the package is recommended)**

```
git clone git@github.com:multicom-toolbox/deepdist.git

cd deepdist
```

**(2) Install and activate python3.6.x environment on Linux (required)**

The installation of python3.6.x may be different for different Linux systems. 
Note: The system is developed and tested under python3.6.x. 

**(3) Make sure bash is installed on the Linux**

For most Linux systems such as Redhat and centos, bash is installed by default. No action is needed.However, because bash is often 
not avaialble at a ubuntu system, you can run the following commands to install it

```
sudo apt-get install csh
sudo ln -s bash /bin/sh.bash 
sudo mv /bin/sh.bash /bin/sh
sudo apt-get install tcsh (if already installed, ignore this)	
```

**(4) Configure DeepDist (required)**

There are two ways to configure the system. One is to use setup_msa.py to configure it to take a multiple sequence alignment (MSA) as input to predict distance map. Another one is to use setup_fasta.py to configure it to take a single protein sequence in the FASTA format to predict distance map. You only need to run one script to configure the sytem. setup_msa.py requires downloading a package of about 30 GB, which is much smaller than 500 GB of setup_fasta.py. If you know how to generate your own MSA, it is better to use setup_msa.py to configure the system 

```
Step 1:
Option 1: python setup_msa.py    # download the light version (recommended)
Option 2: python setup_fasta.py  # downnload the entire package including all the databases and tools for generating MSA and features.

Step 2:
python configure.py

Step 3: 
sh ./installation/set_env.sh
```

**(4) Examples for predicting distance maps using DeepDsit**

<h4>Case 1: run the ensemble of multiple individual deep learning models to predict distance map from a MSA using different commands</h4>

Command 1 for classifying distances into 10 bins
```
sh ./predictors/ensemble/pred_deepdist_msa_construct.sh fasta_file MSA_file output_folder

Examples:

(1)sh ./predictors/ensemble/pred_deepdist_msa_construct.sh ./example/T1019s1.fasta ./example/T1019s1.aln ./predictors/results/T1019s1

(2) On a standard Linux
sh predictors/ensemble/example/pred_deepdist_msa.sh

(3) On Mizzou's Lewis Cluster
sh predictors/ensemble/example/pred_deepdist_msa_lewis.sh
```
Output directory: example/*fasta name*/pred_map_ensem/. The multi_classification distance file (.npy), the real-value distance file (.txt), binary contact file at 8 Angstrom threshold (.txt), binary conctact file (.rr) that can be visualized by [ConEVA](https://github.com/multicom-toolbox/ConEVA). 

Command 2 for classifying distances into 42 bins
```
sh ./predictors/ensemble/pred_deepdist_msa_dist.sh fasta_file MSA_file output_folder
```
Command 3 for different options on deep learning choose
```
python run_deepdist.py -f fasta_file -a MSA_file -o output_dir -m method
```
Different options for -m:

1.mul_class_C: Predict 10-bin multi-classification distance map (The CASP14 official format)

2.mul_class_G: Predict 42-bin multi-classification distance map (It can be convereted trRosetta format. You can replace the distance map in trRosetta npz file by running python ./lib/npy2trRosetta_npz.py -i npy_file -n npz_file -o new_npz_file)

3.mul_label_R: Predict the real-value distance map and the 25-bins multi-classification distance map at the same time. 
	(This is the improved version of DeepDist1)

An example:
```
python run_deepdist.py -f ./example/T1019s1.fasta -a ./example/T1019s1.aln -o ./predictors/results/test/ -m mul_class_C
```

<h4>Case 2: run the ensemble of multiple individual deep learning models to predict distance map from a single sequence in the FASTA format</h4>

```
Command: sh ./predictors/ensemble/pred_deepdist_fasta.sh fasta_file output_folder

Examples of executing the command included in the pacakge: 
On a standard Linux
sh predictors/ensemble/example/pred_deepdist_fasta.sh

On Mizzou's Lewis Cluster:
sh predictors/ensemble/example/pred_deepdist_fasta_lewis.sh
```
The output is stored in the output directory: example/*fasta name*/pred_map_ensem/

<h4>Case 3: Use the deep learning model based on psudo maximum liklihood (plm) feature set to make prediction. </h4>

```
On a standard Linux
sh predictors/individual/pred_deepdist_plm_cpu.sh # See this script for detailed parameters

On Mizzou's Lewis Cluster
sh predictors/individual/pred_deepdist_plm_lewis_cpu.sh
```
The output is stored in the output directory: example/*fasta name*/pred_map0/

Note: The accuracy of the ensemble of multiple deep learning models is generally higher than that of an individual model. 


**(5) Three different ways to generate MSA input for DeepDist**

1.Use DeepMSA to generate a MSA for a protein
Download and install the [DeepMSA](https://zhanglab.dcmb.med.umich.edu/DeepMSA/). This package requires to install large protein sequence databases. 

2.Use [HHblits](https://github.com/soedinglab/hh-suite) to search against a standard protein sequence databse created by HHsuite (e.g. UniRef30) to generate MSA.
The UniRef database created by HHsuite is much smaller than the databases used by DeepMSA. So this approach is faster than DeepMSA, but may be less senstiive for some proteins. For instance, you can download a recent UniRef database (UniRef30_2020_06_hhsuite.tar.gz) here: http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/ as follows. 
```
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
```
Below is an example of generating a MSA from the database.
```
sh ./scripts/hhblits.sh T1049  /Full_path_of_DeeDist/example/T1049.fasta /Full_path_of_DeeDist/predictors/resluts/T1049 /Full_path/UniRef30_2020_06_hhsuite_database
```
3.Use HHblits to search against the Big Fantastic Database (BFD) (https://bfd.mmseqs.com/)
The BFD database is very large. Searching a protein against BFD is slow, but more sensitive. 
Below is an exmaple of generating a MSA from BFD:
```
sh ./scripts/hhblits.sh T1049  /Full_path_of_DeeDist/example/T1049.fasta /Full_path_of_DeeDist/predictors/resluts/T1049 /Full_path_of_BFD_database
```

**(6) Convert a multi-classification distance map into a real-value distance map**

```
python ./lib/mulclass2realdist.py -i input_mulclass_prediction -o output_folder
example
python ./lib/mulclass2realdist.py -i ./example/CASP13_results/mul_class/T0949.npy -o ./predictors/results/1/
```

Note: If you have any further questions, please post your question at this GitHub website or feel free to contact Zhiye Guo: zggc9@umsystem.edu for help.

<h2>References</h2>

1. Guo, Z., Wu, T., Liu, J., Hou, J., & Cheng, J. (2021). Improving deep learning-based protein distance prediction in CASP14. bioRxiv. (https://www.biorxiv.org/content/10.1101/2021.02.02.429462v1.full)
2. Wu, T., Guo, Z., Hou, J., & Cheng, J. (2021). DeepDist: real-value inter-residue distance prediction with deep residual convolutional network. BMC bioinformatics, 22(1), 1-17.(https://link.springer.com/article/10.1186/s12859-021-03960-9)
