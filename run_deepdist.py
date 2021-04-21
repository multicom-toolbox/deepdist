# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:57:26 2020

@author: Zhiye
"""
import sys
import os,glob,re
import time
import subprocess
import argparse
import shutil
from multiprocessing import Process
from random import randint

def is_dir(dirname):
	"""Checks if a path is an actual directory"""
	if not os.path.isdir(dirname):
		msg = "{0} is not a directory".format(dirname)
		raise argparse.ArgumentTypeError(msg)
	else:
		return dirname

def is_file(filename):
	"""Checks if a file is an invalid file"""
	if not os.path.exists(filename):
		msg = "{0} doesn't exist".format(filename)
		raise argparse.ArgumentTypeError(msg)
	else:
		return filename

def chkdirs(fn):
	'''create folder if not exists'''
	dn = os.path.dirname(fn)
	if not os.path.exists(dn): os.makedirs(dn)

		

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.description="DeepDist - Protein real-value/multi-classification distance map predictor."
	parser.add_argument("-f", "--fasta", help="input fasta file",type=is_file,required=True)
	parser.add_argument("-a", "--aln", help="input alignment file",type=is_file,required=True)
	parser.add_argument("-o", "--outdir", help="output folder",type=str,required=True)
	parser.add_argument("-m", "--method", help="model name to load [mul_label_R, mul_class_C, mul_class_G]",type=str, default='mul_class_C', required=False)

	args = parser.parse_args()
	fasta = args.fasta
	aln_file = args.aln
	outdir = args.outdir 
	method = args.method

	GLOABL_Path = sys.path[0]
	env_file = GLOABL_Path + '/installation/path.inf'
	db_tool_dir = open(env_file, 'r').readlines()[1].strip('\n').split('=')[-1]
	if not os.path.exists(db_tool_dir):
		print("Database folder: %s not exists, please check!"%db_tool_dir)
		sys.exit(1)
	chkdirs(outdir + '/')
	#copy fasta to outdir
	os.system('cp %s %s'%(fasta, outdir))
	fasta_file = fasta.split('/')[-1]    
	fasta_name = fasta_file.split('.')[0]
	fasta = os.path.join(outdir, fasta_file) # is the full path of fasta

	vir_env = '%s/env/deepdist_virenv/bin/python'%GLOABL_Path
	models_dir = []
	if method == 'mul_class_G':
		models_dir.append(GLOABL_Path + '/models/pretrain/MULTICOM-DIST/1.dres152_deepcov_cov_ccmpred_pearson_pssm/')
		models_dir.append(GLOABL_Path + '/models/pretrain/MULTICOM-DIST/2.dres152_deepcov_plm_pearson_pssm/')
		models_dir.append(GLOABL_Path + '/models/pretrain/MULTICOM-DIST/3.res152_deepcov_pre_freecontact/')
		models_dir.append(GLOABL_Path + '/models/pretrain/MULTICOM-DIST/4.res152_deepcov_other/')
	elif method == 'mul_class_C':
		models_dir.append(GLOABL_Path + '/models/pretrain/MULTICOM-CONSTRUCT/1.dres152_deepcov_cov_ccmpred_pearson_pssm/')
		models_dir.append(GLOABL_Path + '/models/pretrain/MULTICOM-CONSTRUCT/2.dres152_deepcov_plm_pearson_pssm/')
		models_dir.append(GLOABL_Path + '/models/pretrain/MULTICOM-CONSTRUCT/3.res152_deepcov_pre_freecontact/')
		models_dir.append(GLOABL_Path + '/models/pretrain/MULTICOM-CONSTRUCT/4.res152_deepcov_other/')
	elif method == 'mul_label_R':
		models_dir.append(GLOABL_Path + '/models/pretrain/deepdist_v3rc_AGR_11k/1.dres152_deepcov_cov_ccmpred_pearson_pssm/')
		models_dir.append(GLOABL_Path + '/models/pretrain/deepdist_v3rc_AGR_11k/2.dres152_deepcov_plm_pearson_pssm/')
		models_dir.append(GLOABL_Path + '/models/pretrain/deepdist_v3rc_AGR_11k/3.res152_deepcov_pre_freecontact/')
		models_dir.append(GLOABL_Path + '/models/pretrain/deepdist_v3rc_AGR_11k/4.res152_deepcov_other/')
	print(models_dir)


	script_file = '%s %s/lib/Model_predict_v2.py %s %s %s %s %s %s %s %s %s'%(vir_env, GLOABL_Path, db_tool_dir, fasta, aln_file, 
		models_dir[0], models_dir[1], models_dir[2],models_dir[3],outdir, method)
	os.system(script_file)


