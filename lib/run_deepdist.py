# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:57:26 2020

@author: Zhiye
"""
import sys
import os,glob,re
import time
import numpy as np
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

def run_shell_file(filename):
	outfile = filename.split('.')[0] + '.out'
	if not os.path.exists(filename):
		print("Shell file not exist: %s, please check!"%filename)
		sys.exit(1)
	print("parent %s,child %s,name: %s"%(os.getppid(),os.getpid(),filename))
	os.chdir(os.path.dirname(filename))
	os.system('./%s > %s'%(os.path.basename(filename), outfile))

def cut_domain_fasta(fasta_name, outdir, dm_index_file):
	fasta = open(outdir + '/' + fasta_name + '.fasta', 'r').readlines()[1].strip('\n')
	fl_len = len(fasta)
	dm_name_dict = {}
	f = open(dm_index_file, 'r')
	for line in f.readlines():
		if line == '\n':
			continue
		line_list = line.strip('\n').split(' ')
		index_info = line_list[1]
		# insertion
		dm_index_insert=[]
		if '-' in line_list[2]:
			dm_index_insert = line_list[2].split('-')
		dm_num = index_info.split(':')[0]
		dm_index = index_info.split(':')[-1].split('-')
		dm_fasta_name = fasta_name + '-D' + str(int(dm_num)+1)
		dm_name_dict[dm_fasta_name] = dm_index + dm_index_insert
		dm_fasta_file = outdir + '/' + dm_fasta_name + '.fasta'
		if os.path.exists(dm_fasta_file) and os.path.getsize(dm_fasta_file) != 0:
			continue
		dm_fasta = fasta[int(dm_index[0])-1:int(dm_index[1])]
		if dm_index_insert != []:
			dm_fasta_insert = fasta[int(dm_index_insert[0])-1:int(dm_index_insert[1])]
			dm_fasta = dm_fasta + dm_fasta_insert
		dm_len = len(dm_fasta)
		if dm_len >= fl_len:
			return False
		fisrt_line = '>' + dm_fasta_name + '\n'
		print("Cut domain fasta out: %s"%dm_fasta_name)
		f = open(dm_fasta_file, 'w')
		f.write(fisrt_line)
		f.write(dm_fasta)
		f.close()
		# with open(dm_fasta_file, 'a') as myfile:
		# 	myfile.write(fisrt_line)
		# 	myfile.write(dm_fasta)
	return dm_name_dict

def combine_dm_fl(full_length_dir, domain_dir_dict, dm_name_dict, serve_name, method = 'mul_lable_R'):
	if 'R' in method:
		mul_thred = [0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.0, 15.1]
	elif 'C' in method:
		mul_thred = [0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 18.0]

	if serve_name == None:
		fl_map_folder = full_length_dir + '/ensemble/'
		combine_folder = full_length_dir + '/ensemble/'
	else:
		fl_map_folder = full_length_dir + '/ensemble/' + serve_name
		combine_folder = full_length_dir + '/ensemble/'+ serve_name
	for domain_name in dm_name_dict:
		print('Combine %s'% domain_name)
		full_name = domain_name.split('-')[0]
		if serve_name == None:
			dm_map_folder = domain_dir_dict[domain_name] + '/ensemble/'
		else:
			dm_map_folder = domain_dir_dict[domain_name] + '/ensemble/' + serve_name

		domain_start = int(dm_name_dict[domain_name][0])-1
		domain_end = int(dm_name_dict[domain_name][1])
		insertion_flag = False
		if len(dm_name_dict[domain_name]) > 2: #insertion
			insertion_flag = True
			domain_start_1 = int(dm_name_dict[domain_name][2])-1
			domain_end_1 = int(dm_name_dict[domain_name][3])

		if 'mul_lable' in method  or 'real_dist' in method:
			domain_file = dm_map_folder + '/pred_map_ensem/' + domain_name + '.txt'
			full_file = fl_map_folder + '/pred_map_ensem/'  + full_name +'.txt'
			combine_file = combine_folder + '/pred_map_ensem_dm/' + full_name + '.txt'
			real_dist_file = combine_folder + '/pred_map_ensem_dm/' + '/real_dist/' + full_name + '.txt'
			chkdirs(combine_file)
			chkdirs(real_dist_file)

			if not os.path.exists(domain_file) or not os.path.exists(full_file):
				print("Domain or full map not exists, please check!%s %s"%(domain_file, full_file))
				continue
			# for multi-domain in one map
			if os.path.exists(combine_file):
				full_map = np.loadtxt(combine_file)
			else:
				full_map = np.loadtxt(full_file)
			domain_map = np.loadtxt(domain_file)
			L = full_map.shape[0]
			enmpty_map = np.zeros((L, L))
			if insertion_flag == False:
				enmpty_map[domain_start:domain_end, domain_start:domain_end] = domain_map
				full_map[enmpty_map > full_map] = enmpty_map [enmpty_map > full_map]
			else:
				enmpty_map[domain_start:domain_end, domain_start:domain_end] = domain_map[:(domain_end-domain_start), :(domain_end-domain_start)]
				enmpty_map[domain_start_1:domain_end_1, domain_start_1:domain_end_1] = domain_map[(domain_end-domain_start):, (domain_end-domain_start):]

				enmpty_map[domain_start:domain_end, domain_start_1:domain_end_1] = domain_map[:(domain_end-domain_start), (domain_end-domain_start):]
				enmpty_map[domain_start_1:domain_end_1, domain_start:domain_end] = domain_map[(domain_end-domain_start):, :(domain_end-domain_start)]

				full_map[enmpty_map > full_map] = enmpty_map [enmpty_map > full_map]

			np.savetxt(combine_file, full_map, fmt='%.4f')
			real_dist = 1/full_map
			real_dist[real_dist>100] = 100
			real_dist[real_dist<1] = 1
			np.savetxt(real_dist_file, real_dist, fmt='%.4f')

		# mulclass
		if 'mul_lable' in method or 'mul_class' in method:
			#ensemble mul class and generate dist rr
			if 'mul_lable' in method:
				domain_file = dm_map_folder + '/pred_map_mul_class_ensem/'+ '/mul_class/' + domain_name + '.npy'
				full_file = fl_map_folder + '/pred_map_mul_class_ensem/' + '/mul_class/' + full_name +'.npy'
				combine_file = combine_folder + '/pred_map_mul_class_ensem_dm/' + '/mul_class/' + full_name + '.npy'
			elif 'mul_class' in method:
				domain_file = dm_map_folder + '/pred_map_ensem/'+ '/mul_class/' + domain_name + '.npy'
				full_file = fl_map_folder + '/pred_map_ensem/' + '/mul_class/' + full_name +'.npy'
				combine_file = combine_folder + '/pred_map_ensem_dm/' + '/mul_class/' + full_name + '.npy'
			chkdirs(combine_file)
			if not os.path.exists(domain_file) or not os.path.exists(full_file):
				print("Domain or full map not exists, please check!%s,%s"%(domain_file, full_file))
				continue
			# for multi-domain in one map
			if os.path.exists(combine_file):
				full_map = np.load(combine_file)
			else:
				full_map = np.load(full_file)
			domain_map = np.load(domain_file)
			L = full_map.shape[1]
			C = full_map.shape[-1]
			enmpty_map = np.zeros((1, L, L, C))
			# enmpty_map[0, domain_start:domain_end, domain_start:domain_end, :] = domain_map

			if insertion_flag == False:
				enmpty_map[0, domain_start:domain_end, domain_start:domain_end, :] = domain_map
			else:
				enmpty_map[0, domain_start:domain_end, domain_start:domain_end, :] = domain_map[0, :(domain_end-domain_start), :(domain_end-domain_start), :]
				enmpty_map[0, domain_start_1:domain_end_1, domain_start_1:domain_end_1, :] = domain_map[0, (domain_end-domain_start):, (domain_end-domain_start):, :]

				enmpty_map[0, domain_start:domain_end, domain_start_1:domain_end_1, :] = domain_map[0, :(domain_end-domain_start), (domain_end-domain_start):, :]
				enmpty_map[0, domain_start_1:domain_end_1, domain_start:domain_end, :] = domain_map[0, (domain_end-domain_start):, :(domain_end-domain_start), :]

			if C == 25:
				enmpty_map_bin = enmpty_map[:,:, :, 0:8].sum(axis=-1)
				full_map_bin = full_map[:,:, :, 0:8].sum(axis=-1)
			elif C == 10:
				enmpty_map_bin = enmpty_map[:,:, :, 0:3].sum(axis=-1)
				full_map_bin = full_map[:,:, :, 0:3].sum(axis=-1)


			full_map[enmpty_map_bin > full_map_bin, :] = enmpty_map [enmpty_map_bin > full_map_bin, :]
			# full_map[enmpty_map > full_map] = enmpty_map [enmpty_map > full_map]
			channel_sum = np.sum(full_map, axis = -1)
			new_full = full_map / channel_sum[:,:,:, np.newaxis]
			np.save(combine_file, new_full)

		# mulclass bin from bin avg
		if 'mul_lable' in method or 'mul_class' in method:
			#ensemble mul class and generate dist rr
			if 'mul_lable' in method:
				domain_file = dm_map_folder + '/pred_map_mul_class_ensem/'+ '/bin_map_from_bin_avg/' + domain_name + '.txt'
				full_file = fl_map_folder + '/pred_map_mul_class_ensem/' + '/bin_map_from_bin_avg/' + full_name +'.txt'
				combine_file = combine_folder + '/pred_map_mul_class_ensem_dm/' + '/bin_map_from_bin_avg/' + full_name + '.txt'
			elif 'mul_class' in method:
				domain_file = dm_map_folder + '/pred_map_ensem/'+ '/bin_map_from_bin_avg/' + domain_name + '.txt'
				full_file = fl_map_folder + '/pred_map_ensem/' + '/bin_map_from_bin_avg/' + full_name +'.txt'
				combine_file = combine_folder + '/pred_map_ensem_dm/' + '/bin_map_from_bin_avg/' + full_name + '.txt'
			chkdirs(combine_file)
			if not os.path.exists(domain_file) or not os.path.exists(full_file):
				print("Domain or full map not exists, please check!%s,%s"%(domain_file, full_file))
				continue
			# for multi-domain in one map
			if os.path.exists(combine_file):
				full_map = np.loadtxt(combine_file)
			else:
				full_map = np.loadtxt(full_file)
			domain_map = np.loadtxt(domain_file)
			L = full_map.shape[0]
			enmpty_map = np.zeros((L, L))
			if insertion_flag == False:
				enmpty_map[domain_start:domain_end, domain_start:domain_end] = domain_map
				full_map[enmpty_map > full_map] = enmpty_map [enmpty_map > full_map]
			else:
				enmpty_map[domain_start:domain_end, domain_start:domain_end] = domain_map[:(domain_end-domain_start), :(domain_end-domain_start)]
				enmpty_map[domain_start_1:domain_end_1, domain_start_1:domain_end_1] = domain_map[(domain_end-domain_start):, (domain_end-domain_start):]

				enmpty_map[domain_start:domain_end, domain_start_1:domain_end_1] = domain_map[:(domain_end-domain_start), (domain_end-domain_start):]
				enmpty_map[domain_start_1:domain_end_1, domain_start:domain_end] = domain_map[(domain_end-domain_start):, :(domain_end-domain_start)]

				full_map[enmpty_map > full_map] = enmpty_map [enmpty_map > full_map]
			np.savetxt(combine_file, full_map, fmt='%.4f')

		if 'mul_lable' in method or 'mul_class' in method:
			#ensemble mul class and generate dist rr
			if 'mul_lable' in method:
				mul_class_ensemble_dir = combine_folder + '/pred_map_mul_class_ensem_dm/'
			elif 'mul_class' in method:
				mul_class_ensemble_dir = combine_folder + '/pred_map_ensem_dm/'
			sum_cmap_dir = mul_class_ensemble_dir + '/mul_class/'
			bin_dir = mul_class_ensemble_dir + '/bin_map/'
			dist_dir = mul_class_ensemble_dir + '/real_dist/'
			score_dir = mul_class_ensemble_dir + '/score/'
			dev_dir = mul_class_ensemble_dir + '/dev_np/'
			rr_folder = dist_dir + '/dist_rr/'
			chkdirs(mul_class_ensemble_dir)
			chkdirs(sum_cmap_dir)
			chkdirs(bin_dir)
			chkdirs(dist_dir)
			chkdirs(rr_folder)
			chkdirs(score_dir)
			chkdirs(dev_dir)

			seq_name = full_name
			print('transform mulclass npy into dist,deviation,score ', seq_name)
			mul_sum_map = sum_cmap_dir + seq_name + '.npy'
			bin_sum_map = bin_dir + seq_name + '.txt'
			dist_sum_map = dist_dir + seq_name + '.txt'
			dev_file = dev_dir + '/' + seq_name + '.txt'
			score_file = score_dir + '/' + seq_name + '.txt'

			mul_class = np.load(mul_sum_map)
			L = mul_class.shape[1]
			_class = mul_class.shape[-1]
			mul_class_weighted = np.zeros((L, L, _class))
			for i in range(_class):
				mul_class_single = np.copy(mul_class[0, :, :, i])
				mul_class_single *= mul_thred[i]
				mul_class_weighted[:, :, i] = mul_class_single
			dist_from_mulclass = mul_class_weighted.sum(axis=-1)
			dist_from_mulclass = (dist_from_mulclass + dist_from_mulclass.T) / 2.0  # this is avg of mul class

			mul_class_dev = np.std(mul_class_weighted, axis=-1)

			if len(mul_thred) == 25:
				bin_from_mul = mul_class[0,:, :, 0:8].sum(axis=-1).reshape(L, L)
				score = mul_class[0,:, :, 0:22].sum(axis=-1).reshape(L, L)
			elif len(mul_thred) == 10:
				bin_from_mul = mul_class[0,:, :, 0:3].sum(axis=-1).reshape(L, L)
				score = mul_class[0,:, :, 0:7].sum(axis=-1).reshape(L, L)

			np.savetxt(dist_sum_map, dist_from_mulclass, fmt='%.4f')
			np.savetxt(dev_file, mul_class_dev, fmt='%.4f')
			np.savetxt(bin_sum_map, bin_from_mul, fmt='%.4f')
			np.savetxt(score_file, score, fmt='%.4f')
	if 'mul_lable' in method:
		real_dist_dm_dir = combine_folder + '/pred_map_ensem_dm/'
		mul_class_dm_dir = combine_folder + '/pred_map_mul_class_ensem_dm/'
	elif 'mul_class' in method:
		real_dist_dm_dir = combine_folder + '/pred_map_ensem_dm/'
		mul_class_dm_dir = combine_folder + '/pred_map_ensem_dm/'
	return real_dist_dm_dir, mul_class_dm_dir

def ensemble_aln_msa(fasta_name, outdir, serve_name, method = 'mul_lable_R'):
	if serve_name == None:
		model_dir = [outdir + '/aln/', outdir + '/msa/']
		ensemble_dir = outdir + '/ensemble/'
	else:
		model_dir = [outdir + '/aln/' + serve_name, outdir + '/msa/' + serve_name]
		ensemble_dir = outdir + '/ensemble/' + serve_name
	model_num = len(model_dir)
	chkdirs(ensemble_dir)
	if 'R' in method:
		mul_thred = [0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.0, 15.1]
	elif 'C' in method:
		mul_thred = [0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 18.0]

	if 'mul_lable' in method:
		ensemble_rd_dir =  '/pred_map_ensem/'
		ensemble_mc_dir =  '/pred_map_mul_class_ensem/'
	else:
		ensemble_rd_dir =  '/pred_map_ensem/'
		ensemble_mc_dir =  '/pred_map_ensem/'

	real_dist_ensemble_dir = ensemble_dir + ensemble_rd_dir
	if 'mul_lable' in method or 'real_dist' in method:
		sum_cmap_dir = real_dist_ensemble_dir
		sum_dmap_dir = sum_cmap_dir + '/real_dist/'
		rr_folder = sum_dmap_dir + '/dist_rr/'
		chkdirs(sum_cmap_dir)
		chkdirs(sum_dmap_dir)
		chkdirs(rr_folder)

		seq_name = fasta_name
		print('ensemble real dist ', seq_name)

		sum_map_filename = sum_cmap_dir + seq_name + '.txt'
		sum_map = 0
		for i in range(model_num):
			cmap_file = model_dir[i] + ensemble_rd_dir + seq_name + '.txt'
			cmap = np.loadtxt(cmap_file, dtype=np.float32)
			sum_map += cmap
		sum_map /= model_num
		np.savetxt(sum_map_filename, sum_map, fmt='%.4f')
		real_dist_file = sum_dmap_dir + '/' + seq_name + '.txt'
		real_dist = 1/sum_map
		real_dist[real_dist>100] = 100
		real_dist[real_dist<1] = 1
		np.savetxt(real_dist_file, real_dist, fmt='%.4f')
	mul_class_ensemble_dir = ensemble_dir + ensemble_mc_dir
	if 'mul_lable' in method or 'mul_class' in method:
		#ensemble mul class and generate dist rr
		sum_cmap_dir = mul_class_ensemble_dir + '/mul_class/'
		bin_dir = mul_class_ensemble_dir + '/bin_map/'
		bin_from_bin_dir = mul_class_ensemble_dir + '/bin_map_from_bin_avg/'
		dist_dir = mul_class_ensemble_dir + '/real_dist/'
		score_dir = mul_class_ensemble_dir + '/score/'
		dev_dir = mul_class_ensemble_dir + '/dev_np/'
		chkdirs(mul_class_ensemble_dir)
		chkdirs(sum_cmap_dir)
		chkdirs(bin_dir)
		chkdirs(bin_from_bin_dir)
		chkdirs(dist_dir)
		chkdirs(score_dir)
		chkdirs(dev_dir)

		seq_name = fasta_name
		print('ensemble mul class ', seq_name)

		mul_sum_map = sum_cmap_dir + seq_name + '.npy'
		bin_sum_map = bin_dir + seq_name + '.txt'
		bin_from_bin = bin_from_bin_dir + seq_name + '.txt'
		dist_sum_map = dist_dir + seq_name + '.txt'
		dev_file = dev_dir + '/' + seq_name + '.txt'
		score_file = score_dir + '/' + seq_name + '.txt'

		sum_map = 0
		mul_class = 0
		for i in range(model_num):
			cmap_file = model_dir[i] + ensemble_mc_dir + seq_name + '.txt'
			cmap = np.loadtxt(cmap_file, dtype=np.float32)
			sum_map += cmap
			npy_file = model_dir[i] + ensemble_mc_dir + '/mul_class/' + seq_name + '.npy'
			npy = np.load(npy_file)
			mul_class += npy

		sum_map /= model_num
		mul_class /= model_num
		L = mul_class.shape[1]
		_class = mul_class.shape[-1]
		mul_class_weighted = np.zeros((L, L, _class))
		for i in range(_class):
			mul_class_single = np.copy(mul_class[0, :, :, i])
			mul_class_single *= mul_thred[i]
			mul_class_weighted[:, :, i] = mul_class_single
		dist_from_mulclass = mul_class_weighted.sum(axis=-1)
		dist_from_mulclass = (dist_from_mulclass + dist_from_mulclass.T) / 2.0  # this is avg of mul class

		mul_class_dev = np.std(mul_class_weighted, axis=-1)
		if len(mul_thred) == 25:
			bin_from_mul = mul_class[0,:, :, 0:8].sum(axis=-1).reshape(L, L)
			score = mul_class[0,:, :, 0:22].sum(axis=-1).reshape(L, L)
		elif len(mul_thred) == 10:
			bin_from_mul = mul_class[0,:, :, 0:3].sum(axis=-1).reshape(L, L)
			score = mul_class[0,:, :, 0:7].sum(axis=-1).reshape(L, L)

		np.save(mul_sum_map, mul_class)
		np.savetxt(dist_sum_map, dist_from_mulclass, fmt='%.4f')
		np.savetxt(dev_file, mul_class_dev, fmt='%.4f')
		np.savetxt(bin_sum_map, bin_from_mul, fmt='%.4f')
		np.savetxt(bin_from_bin, sum_map, fmt='%.4f')
		np.savetxt(score_file, score, fmt='%.4f')
	return real_dist_ensemble_dir, mul_class_ensemble_dir

def generate_distrr_for_dfold(fasta_name, outdir, real_dist_dir, mul_class_dir, method = 'mul_lable_R'):
	#real dist dir is the final distance map, mul class dir contain final dist map, dev map, score from mul class
	real_dist_rr_folder = real_dist_dir + '/real_dist/dist_rr/'
	if os.path.isdir(real_dist_dir) == True:
		chkdirs(real_dist_rr_folder)
		dist_rr_file = real_dist_rr_folder + fasta_name + '.dist.rr'
		if os.path.exists(dist_rr_file):
			os.remove(dist_rr_file)
		fasta = open(outdir + '/' + fasta_name + '.fasta', 'r').readlines()[1].strip('\n')
		with open(dist_rr_file, "a") as myfile:
			myfile.write(str(fasta))
			myfile.write('\n')
		dist = np.loadtxt(real_dist_dir + '/real_dist/' + fasta_name + '.txt')
		L = dist.shape[0]
		for i in range(0, L):
			for j in range(i, L):  # for confold2, j = i+5 for dfold j= i+2
				if dist[i][j] > 16:
					continue
				str_to_write = str(i + 1) + " " + str(j + 1) + " 0 " + str(dist[i][j]) + ' 0.1 0.1 ' + str(1/(dist[i][j]+0.0000000001)) + "\n"
				with open(dist_rr_file, "a") as myfile:
					myfile.write(str_to_write)
	mul_class_rr_folder = mul_class_dir + '/real_dist/dist_rr/'
	if os.path.isdir(mul_class_dir) == True:
		chkdirs(mul_class_rr_folder)
		dist_rr_file = mul_class_rr_folder + fasta_name + '.dist.rr'
		if os.path.exists(dist_rr_file):
			os.remove(dist_rr_file)
		fasta = open(outdir + '/' + fasta_name + '.fasta', 'r').readlines()[1].strip('\n')
		with open(dist_rr_file, "a") as myfile:
			myfile.write(str(fasta))
			myfile.write('\n')
		dist_from_mulclass = np.loadtxt(mul_class_dir + '/real_dist/' + fasta_name + '.txt')
		mul_class_dev =  np.loadtxt(mul_class_dir + '/dev_np/' + fasta_name + '.txt')
		score =  np.loadtxt(mul_class_dir + '/score/' + fasta_name + '.txt')
		L = dist_from_mulclass.shape[0]
		for i in range(0, L):
			for j in range(i, L):  # for confold2, j = i+5
				str_to_write = str(i + 1) + " " + str(j + 1) + " 0 " + str(dist_from_mulclass[i][j])+ ' ' + str(mul_class_dev[i][j]) + ' ' + str(mul_class_dev[i][j]) + ' ' + str(score[i][j]) + "\n"
				with open(dist_rr_file, "a") as myfile:
					myfile.write(str_to_write)
	return real_dist_rr_folder, mul_class_rr_folder

def prob_map2sort_rr(prob_map):
	L = prob_map.shape[0]
	rr_list = []
	for i in range(L):
		for j in range(i+1,L):
			rr_list.append([i,j,prob_map[i,j]])
	rr_list = np.array(rr_list)
	sort_rr = rr_list[np.argsort(-rr_list[:,2])]
	return sort_rr

def prob_map2sort_rr_bin(prob_map):
	L = prob_map.shape[0]
	rr_list = []
	for i in range(L):
		for j in range(i+5,L):
			if prob_map[i,j] > 1:
				prob_map[i,j]=1
			rr_list.append([i,j,prob_map[i,j]])
	rr_list = np.array(rr_list)
	sort_rr = rr_list[np.argsort(-rr_list[:,2])]
	return sort_rr

# MULTICOM-CLUSTER@missouri.edu MULTICOM-CONSTRUCT@missouri.edu MULTICOM-DIST@missouri.edu MULTICOM-HYBRID@missouri.edu MULTICOM-DEEP@missouri.edu
def generate_distrr_for_dist(fasta_name, outdir, mul_class_dir, serve_name = 'serve1'):
	mul_class_rr_folder = mul_class_dir + '/mul_class/rr/'
	if os.path.isdir(mul_class_dir) == True:
		chkdirs(mul_class_rr_folder)
		dist_rr_file = mul_class_rr_folder + fasta_name + '.rr'
		if os.path.exists(dist_rr_file):
			os.remove(dist_rr_file)
		fasta = open(outdir + '/' + fasta_name + '.fasta', 'r').readlines()[1].strip('\n')
		with open(dist_rr_file, "a") as myfile:
			myfile.write("PFRMAT RR\n")
			myfile.write("TARGET %s\n"%fasta_name)
			if serve_name == 'serve1':
				myfile.write("AUTHOR MULTICOM-CONSTRUCT\n")
			elif serve_name == 'serve2':
				myfile.write("AUTHOR MULTICOM-DIST\n")
			elif serve_name == 'serve3':
				myfile.write("AUTHOR MULTICOM-AI\n")
			elif serve_name == 'serve4':
				myfile.write("AUTHOR MULTICOM-HYBRID\n")
			elif serve_name == 'serve5':
				myfile.write("AUTHOR MULTICOM-DEEP\n")
			myfile.write("REMARK None\n")
			myfile.write("METHOD MULCLASS\n")
			myfile.write("METHOD MULCLASS\n")
			myfile.write("RMODE  2\n")
			myfile.write("MODEL  1\n")
			# myfile.write("".join([fasta[i:i+50]+'\n' for i in range(0, len(fasta), 50)]))

		mulclass = np.load(mul_class_dir + '/mul_class/' + fasta_name + '.npy')
		mulclass = mulclass.squeeze()
		L = mulclass.shape[0]
		C = mulclass.shape[-1] #3.5-19:C=33, 2-22:C=42, 4-20:C=10, 4.5-16:C=25
		if C == 33:
			P1 = np.round(mulclass[:,:,0:2].sum(axis=-1),decimals=3)
			P2 = np.round(mulclass[:,:,2:6].sum(axis=-1),decimals=3)
			P3 = np.round(mulclass[:,:,6:10].sum(axis=-1),decimals=3)
			P4 = np.round(mulclass[:,:,10:14].sum(axis=-1),decimals=3)
			P5 = np.round(mulclass[:,:,14:18].sum(axis=-1),decimals=3)
			P6 = np.round(mulclass[:,:,18:22].sum(axis=-1),decimals=3)
			P7 = np.round(mulclass[:,:,22:26].sum(axis=-1),decimals=3)
			P8 = np.round(mulclass[:,:,26:30].sum(axis=-1),decimals=3)
			P9 = np.round(mulclass[:,:,30:32].sum(axis=-1),decimals=3)
			# P10= np.round(mulclass[:,:,32],decimals=3)
			P10= np.round(1-P1-P2-P3-P4-P5-P6-P7-P8-P9,decimals=3)
			P0 = np.round(P1+P2+P3,decimals=3)
		elif C==42:
			P1 = np.round(mulclass[:,:,0:5].sum(axis=-1),decimals=3)
			P2 = np.round(mulclass[:,:,5:9].sum(axis=-1),decimals=3)
			P3 = np.round(mulclass[:,:,9:13].sum(axis=-1),decimals=3)
			P4 = np.round(mulclass[:,:,13:17].sum(axis=-1),decimals=3)
			P5 = np.round(mulclass[:,:,17:21].sum(axis=-1),decimals=3)
			P6 = np.round(mulclass[:,:,21:25].sum(axis=-1),decimals=3)
			P7 = np.round(mulclass[:,:,25:29].sum(axis=-1),decimals=3)
			P8 = np.round(mulclass[:,:,29:33].sum(axis=-1),decimals=3)
			P9 = np.round(mulclass[:,:,33:37].sum(axis=-1),decimals=3)
			# P10= np.round(mulclass[:,:,37:].sum(axis=-1),decimals=3)
			P10= np.round(1-P1-P2-P3-P4-P5-P6-P7-P8-P9,decimals=3)
			P0 = np.round(P1+P2+P3,decimals=3)
		elif C==38:
			P1 = np.round(mulclass[:,:,0:5].sum(axis=-1),decimals=3)
			P2 = np.round(mulclass[:,:,5:9].sum(axis=-1),decimals=3)
			P3 = np.round(mulclass[:,:,9:13].sum(axis=-1),decimals=3)
			P4 = np.round(mulclass[:,:,13:17].sum(axis=-1),decimals=3)
			P5 = np.round(mulclass[:,:,17:21].sum(axis=-1),decimals=3)
			P6 = np.round(mulclass[:,:,21:25].sum(axis=-1),decimals=3)
			P7 = np.round(mulclass[:,:,25:29].sum(axis=-1),decimals=3)
			P8 = np.round(mulclass[:,:,29:33].sum(axis=-1),decimals=3)
			P9 = np.round(mulclass[:,:,33:37].sum(axis=-1),decimals=3)
			# P10= np.round(mulclass[:,:,37:].sum(axis=-1),decimals=3)
			P10= np.round(1-P1-P2-P3-P4-P5-P6-P7-P8-P9,decimals=3)
			P0 = np.round(P1+P2+P3,decimals=3)
		elif C==10:
			P1 = np.round(mulclass[:,:,0],decimals=3)
			P2 = np.round(mulclass[:,:,1],decimals=3)
			P3 = np.round(mulclass[:,:,2],decimals=3)
			P4 = np.round(mulclass[:,:,3],decimals=3)
			P5 = np.round(mulclass[:,:,4],decimals=3)
			P6 = np.round(mulclass[:,:,5],decimals=3)
			P7 = np.round(mulclass[:,:,6],decimals=3)
			P8 = np.round(mulclass[:,:,7],decimals=3)
			P9 = np.round(mulclass[:,:,8],decimals=3)
			# P10= np.round(mulclass[:,:,9],decimals=3)
			P10= np.round(1-P1-P2-P3-P4-P5-P6-P7-P8-P9,decimals=3)
			P0 = np.round(P1+P2+P3,decimals=3)
		elif C== 25:
			P1 = np.round(mulclass[:,:,0],decimals=3)
			P2 = np.round(mulclass[:,:,1:4].sum(axis=-1),decimals=3)
			P3 = np.round(mulclass[:,:,4:8].sum(axis=-1),decimals=3)
			P4 = np.round(mulclass[:,:,8:12].sum(axis=-1),decimals=3)
			P5 = np.round(mulclass[:,:,12:16].sum(axis=-1),decimals=3)
			P6 = np.round(mulclass[:,:,16:20].sum(axis=-1),decimals=3)
			P7 = np.round(mulclass[:,:,20:24].sum(axis=-1),decimals=3)
			P8 = np.round(1-P1-P2-P3-P4-P5-P6-P7-P8-P9,decimals=3)
			P9 = np.round(1-P1-P2-P3-P4-P5-P6-P7-P8-P9,decimals=3)
			# P10= np.round(mulclass[:,:,24],decimals=3)
			P10= np.round(1-P1-P2-P3-P4-P5-P6-P7-P8-P9,decimals=3)
			P0 = np.round(P1+P2+P3,decimals=3)
		P0 = np.round(P0,decimals=3)
		if np.isnan(P0.any()) == True:
			print("Nan error! please check ccmpred!")
			sys.exit(1)
		P0[P0 >= 1.0] = 1.0
		P10[P10 <= 0.0] = 0.0
		P0_sort = prob_map2sort_rr(P0)
		for l in range(len(P0_sort)):
			i = int(P0_sort[l,0])
			j = int(P0_sort[l,1])
			P0 = np.round(P0_sort[l,2], decimals=3)
			str_to_write = str(i + 1) + " " + str(j + 1) + " " + str(P0) + " " + str(P1[i, j]) + " " + str(P2[i, j]) + " " + str(P3[i, j]) + " " + str(P4[i, j]) \
			+ " " + str(P5[i, j]) + " " + str(P6[i, j]) + " " + str(P7[i, j]) + " " + str(P8[i, j]) + " " + str(P9[i, j]) + " " + str(P10[i, j]) + "\n"
			with open(dist_rr_file, "a") as myfile:
				myfile.write(str_to_write)
		# for i in range(0, L):
		# 	for j in range(i + 1, L):  # for confold2, j = i+5
		# 		str_to_write = str(i + 1) + " " + str(j + 1) + " " + str(P0[i, j]) + " " + str(P1[i, j]) + " " + str(P2[i, j]) + " " + str(P3[i, j]) + " " + str(P4[i, j]) \
		# 		+ " " + str(P5[i, j]) + " " + str(P6[i, j]) + " " + str(P7[i, j]) + " " + str(P8[i, j]) + " " + str(P9[i, j]) + " " + str(P10[i, j]) + "\n"
		# 		with open(dist_rr_file, "a") as myfile:
		# 			myfile.write(str_to_write)
		with open(dist_rr_file, "a") as myfile:
			myfile.write("END\n")
	return mul_class_rr_folder

def generate_distrr_for_contact(fasta_name, outdir, real_dist_dir, mul_class_dir, method = 'mul_lable_R', serve_name = 'serve1'):
	real_dist_rr_folder = real_dist_dir + '/rr/'
	if 'mul_lable' in method:
		if os.path.isdir(real_dist_dir) == True:
			chkdirs(real_dist_rr_folder)
			dist_rr_file = real_dist_rr_folder + fasta_name + '.rr'
			if os.path.exists(dist_rr_file):
				os.remove(dist_rr_file)
			fasta = open(outdir + '/' + fasta_name + '.fasta', 'r').readlines()[1].strip('\n')
			with open(dist_rr_file, "a") as myfile:
				myfile.write("PFRMAT RR\n")
				myfile.write("TARGET %s\n"%fasta_name)
				myfile.write("AUTHOR MULTICOM-CLUSTER\n")
				myfile.write("REMARK None\n")
				myfile.write("METHOD BINARY\n")
				myfile.write("METHOD BINARY\n")
				myfile.write("RMODE  1\n")
				myfile.write("MODEL  1\n")
				# myfile.write("".join([fasta[i:i+50]+'\n' for i in range(0, len(fasta), 50)]))
			dist = np.loadtxt(real_dist_dir + fasta_name + '.txt')
			L = dist.shape[0]
			P0_sort = prob_map2sort_rr_bin(dist)
			for l in range(len(P0_sort)):
				i = int(P0_sort[l,0])
				j = int(P0_sort[l,1])
				P0 = np.round(P0_sort[l,2],decimals=3)
				str_to_write = str(i + 1) + " " + str(j + 1) + " " + str(P0) + "\n"
				with open(dist_rr_file, "a") as myfile:
					myfile.write(str_to_write)
			with open(dist_rr_file, "a") as myfile:
				myfile.write("END\n")
	if 'mul_lable' in method or 'mul_class' in method:
		mul_class_rr_folder = mul_class_dir + '/bin_map_from_bin_avg/rr/'
		if os.path.isdir(mul_class_dir) == True:
			chkdirs(mul_class_rr_folder)
			dist_rr_file = mul_class_rr_folder + fasta_name + '.rr'
			if os.path.exists(dist_rr_file):
				os.remove(dist_rr_file)
			fasta = open(outdir + '/' + fasta_name + '.fasta', 'r').readlines()[1].strip('\n')
			with open(dist_rr_file, "a") as myfile:
				myfile.write("PFRMAT RR\n")
				myfile.write("TARGET %s\n"%fasta_name)
				myfile.write("AUTHOR MULTICOM-CLUSTER\n")
				myfile.write("REMARK None\n")
				myfile.write("METHOD BINARY\n")
				myfile.write("METHOD BINARY\n")
				myfile.write("RMODE  1\n")
				myfile.write("MODEL  1\n")
				# myfile.write("".join([fasta[i:i+50]+'\n' for i in range(0, len(fasta), 50)]))
			dist_from_mulclass = np.loadtxt(mul_class_dir + '/bin_map_from_bin_avg/' + fasta_name + '.txt')
			L = dist_from_mulclass.shape[0]
			P0_sort = prob_map2sort_rr_bin(dist_from_mulclass)
			for l in range(len(P0_sort)):
				i = int(P0_sort[l,0])
				j = int(P0_sort[l,1])
				P0 = np.round(P0_sort[l,2],decimals=3)
				str_to_write = str(i + 1) + " " + str(j + 1) + " " + str(P0) + "\n"
				with open(dist_rr_file, "a") as myfile:
					myfile.write(str_to_write)
			with open(dist_rr_file, "a") as myfile:
				myfile.write("END\n")
		return real_dist_rr_folder, mul_class_rr_folder

def generate_distrr_for_coneva(fasta_name, outdir, real_dist_dir, mul_class_dir, method = 'mul_lable_R'):
	real_dist_rr_folder = real_dist_dir + '/rr/'
	if 'mul_lable' in method:
		if os.path.isdir(real_dist_dir) == True:
			chkdirs(real_dist_rr_folder)
			dist_rr_file = real_dist_rr_folder + fasta_name + '.rr'
			if os.path.exists(dist_rr_file):
				os.remove(dist_rr_file)
			fasta = open(outdir + '/' + fasta_name + '.fasta', 'r').readlines()[1].strip('\n')
			with open(dist_rr_file, "a") as myfile:
				myfile.write(fasta + '\n')
				# myfile.write("".join([fasta[i:i+50]+'\n' for i in range(0, len(fasta), 50)]))
			dist = np.loadtxt(real_dist_dir + fasta_name + '.txt')
			L = dist.shape[0]
			for i in range(0, L):
				for j in range(i + 1, L):  # for confold2, j = i+5 for dfold j= i+2
					str_to_write = str(i + 1) + " " + str(j + 1) + " 0 8 " + str(np.round(dist[i][j],decimals=4)) + "\n"
					with open(dist_rr_file, "a") as myfile:
						myfile.write(str_to_write)
			with open(dist_rr_file, "a") as myfile:
				myfile.write("END\n")
	if 'mul_lable' in method or 'mul_class' in method:
		mul_class_rr_folder = mul_class_dir + '/bin_map_from_bin_avg/rr/'
		if os.path.isdir(mul_class_dir) == True:
			chkdirs(mul_class_rr_folder)
			dist_rr_file = mul_class_rr_folder + fasta_name + '.rr'
			if os.path.exists(dist_rr_file):
				os.remove(dist_rr_file)
			fasta = open(outdir + '/' + fasta_name + '.fasta', 'r').readlines()[1].strip('\n')
			with open(dist_rr_file, "a") as myfile:
				myfile.write(fasta + '\n')
				# myfile.write("".join([fasta[i:i+50]+'\n' for i in range(0, len(fasta), 50)]))
			dist_from_mulclass = np.loadtxt(mul_class_dir + '/bin_map_from_bin_avg/' + fasta_name + '.txt')
			L = dist_from_mulclass.shape[0]
			for i in range(0, L):
				for j in range(i + 1, L):  # for confold2, j = i+5 for dfold j= i+2
					str_to_write = str(i + 1) + " " + str(j + 1) + " 0 8 " + str(np.round(dist[i][j],decimals=4)) + "\n"
					with open(dist_rr_file, "a") as myfile:
						myfile.write(str_to_write)
			with open(dist_rr_file, "a") as myfile:
				myfile.write("END\n")
		return real_dist_rr_folder, mul_class_rr_folder


def generate_pred_shell(shell_file, customdir, outdir, fasta, option, serve_name, model_select = 'mul_lable_R'):
	chkdirs(shell_file)
	# every will remove the old shell file
	if os.path.exists(shell_file):
		os.remove(shell_file)
	with open(shell_file, "a") as myfile:
		myfile.write('#!/bin/bash -l\n')
		myfile.write('export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n')
		myfile.write('export HDF5_USE_FILE_LOCKING=FALSE\n')
		myfile.write('\n##GLOBAL_FLAG\n')
		myfile.write('global_dir=/data/casp14/DeepDist')
		myfile.write('\n## ENV_FLAG\n')
		myfile.write('source $global_dir/env/deepdist_virenv/bin/activate\n')
		if model_select == 'mul_lable_R':
			if option == 'ALN':
				myfile.write('models_dir[0]=$global_dir/models/pretrain/deepdist_v3rc_GR/1.dres152_deepcov_cov_ccmpred_pearson_pssm/\n')
				myfile.write('models_dir[1]=$global_dir/models/pretrain/deepdist_v3rc_GR/2.dres152_deepcov_plm_pearson_pssm/\n')
				myfile.write('models_dir[2]=$global_dir/models/pretrain/deepdist_v3rc_GR/3.res152_deepcov_pre_freecontact/\n')
				myfile.write('models_dir[3]=$global_dir/models/pretrain/deepdist_v3rc_GR/4.res152_deepcov_other/\n')
				myfile.write('output_dir=%s/aln/\n'%(customdir))
			elif option == 'MSA':
				myfile.write('models_dir[0]=$global_dir/models/pretrain/deepdist_v3rc_msa_GR/1.dres152_deepcov_cov_ccmpred_pearson_pssm/\n')
				myfile.write('models_dir[1]=$global_dir/models/pretrain/deepdist_v3rc_msa_GR/2.dres152_deepcov_plm_pearson_pssm/\n')
				myfile.write('models_dir[2]=$global_dir/models/pretrain/deepdist_v3rc_msa_GR/3.res152_deepcov_pre_freecontact/\n')
				myfile.write('models_dir[3]=$global_dir/models/pretrain/deepdist_v3rc_msa_GR/4.res152_deepcov_other/\n')
				myfile.write('output_dir=%s/msa/\n'%(customdir))
		elif model_select == 'mul_class_C_11k':
			if option == 'ALN':
				myfile.write('models_dir[0]=$global_dir/models/pretrain/MULTICOM-CONSTRUCT/1.dres152_deepcov_cov_ccmpred_pearson_pssm/\n')
				myfile.write('models_dir[1]=$global_dir/models/pretrain/MULTICOM-CONSTRUCT/2.dres152_deepcov_plm_pearson_pssm/\n')
				myfile.write('models_dir[2]=$global_dir/models/pretrain/MULTICOM-CONSTRUCT/3.res152_deepcov_pre_freecontact/\n')
				myfile.write('models_dir[3]=$global_dir/models/pretrain/MULTICOM-CONSTRUCT/4.res152_deepcov_other/\n')
				myfile.write('output_dir=%s/aln/\n'%(customdir))
			elif option == 'MSA':
				myfile.write('models_dir[0]=$global_dir/models/pretrain/MULTICOM-CONSTRUCT/1.dres152_deepcov_cov_ccmpred_pearson_pssm/\n')
				myfile.write('models_dir[1]=$global_dir/models/pretrain/MULTICOM-CONSTRUCT/2.dres152_deepcov_plm_pearson_pssm/\n')
				myfile.write('models_dir[2]=$global_dir/models/pretrain/MULTICOM-CONSTRUCT/3.res152_deepcov_pre_freecontact/\n')
				myfile.write('models_dir[3]=$global_dir/models/pretrain/MULTICOM-CONSTRUCT/4.res152_deepcov_other/\n')
				myfile.write('output_dir=%s/msa/\n'%(customdir))
	
		myfile.write('fasta=%s/%s.fasta\n'%(outdir, fasta))
		myfile.write('\n## DBTOOL_FLAG\n')
		myfile.write('db_tool_dir=/data/casp14/com_db_tools/')
		myfile.write('\nprintf \"$global_dir\"\n')
		myfile.write('#################CV_dir output_dir dataset database_path\n')
		if model_select == 'mul_lable_R':
			myfile.write('python $global_dir/lib/Model_predict.py $db_tool_dir $fasta ${models_dir[@]} $output_dir \'mul_lable_R\' %s %s\n'%(option, serve_name))
		elif model_select == 'mul_class_C':
			myfile.write('python $global_dir/lib/Model_predict.py $db_tool_dir $fasta ${models_dir[@]} $output_dir \'mul_class_C\' %s %s\n'%(option, serve_name))
		

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.description="DeepDist - The best real-value distance predictor in the world."
	parser.add_argument("-f", "--fasta", help="input fasta file",type=is_file,required=True)
	parser.add_argument("-o", "--outdir", help="output folder",type=str,required=True)
	parser.add_argument("-dm", "--domain", help="e.g. domain 0:1-124 easy", required=False)
	parser.add_argument("-df", "--dfold", help="0 for dfold rr, 1 for dist rr, 2 for binary rr",type=int, default=0, required=False)
	parser.add_argument("-m", "--method", help="model name to load [mul_lable_R]",type=str, default='mul_lable_R', required=False)
	parser.add_argument("-s", "--serve", help="serve name for different model. e.g. serve1", default=None, required=False)

	args = parser.parse_args()
	fasta = args.fasta
	outdir = args.outdir
	dm_index_file = args.domain
	DF_FLAG = args.dfold
	method = args.method
	serve_name = args.serve
	print(serve_name)

	DM_FLAG = False
	chkdirs(outdir + '/')
	#copy fasta to outdir
	os.system('cp %s %s'%(fasta, outdir))
	fasta_file = fasta.split('/')[-1]
	fasta_name = fasta_file.split('.')[0]
	fasta = os.path.join(outdir, fasta_file) # is the full path of fasta

	#process if have domian info
	if dm_index_file is not None:
		if os.path.exists(dm_index_file) and os.path.getsize(dm_index_file) != 0:
			DM_FLAG = True
			dm_name_dict = cut_domain_fasta(fasta_name, outdir, dm_index_file)
			if dm_name_dict == False:
				print("No domain been detected!")
				DM_FLAG = False
			elif len(dm_name_dict) < 1:
				print("Cut domain failed, please check!")

	#generate shell file
	full_length_dir = outdir + '/full_length/'
	chkdirs(full_length_dir)
	aln_shell_file =  outdir + '/shell/%s_aln.sh'%fasta_name
	msa_shell_file =  outdir + '/shell/%s_msa.sh'%fasta_name
	generate_pred_shell(aln_shell_file, full_length_dir, outdir, fasta_name, 'ALN', serve_name, model_select = method)
	generate_pred_shell(msa_shell_file, full_length_dir, outdir, fasta_name, 'MSA', serve_name, model_select = method)
	os.system('chmod 777 %s'%aln_shell_file)
	os.system('chmod 777 %s'%msa_shell_file)

	if DM_FLAG:
		domain_dir_dict = {}
		count = 0
		for dm_fasta_name in dm_name_dict:
			domain_dir = outdir + '/%s/'%dm_fasta_name
			count += 1
			chkdirs(domain_dir)
			domain_dir_dict[dm_fasta_name] = domain_dir
			aln_shell_file =  outdir + '/shell/%s_aln.sh'%dm_fasta_name
			msa_shell_file =  outdir + '/shell/%s_msa.sh'%dm_fasta_name
			generate_pred_shell(aln_shell_file, domain_dir, outdir, dm_fasta_name, 'ALN', serve_name, model_select = method)
			generate_pred_shell(msa_shell_file, domain_dir, outdir, dm_fasta_name, 'MSA', serve_name, model_select = method)
			os.system('chmod 777 %s'%aln_shell_file)
			os.system('chmod 777 %s'%msa_shell_file)

	print('Subprocess the deepaln and deepmsa pipline!')
	procs = []
	for file in glob.glob(outdir + '/shell/%s*.sh'%fasta_name):
		proc = Process(target=run_shell_file, args=(file,))
		procs.append(proc)
		proc.start()

	for proc in procs:
		proc.join()

	# ensemble
	real_dist_dir, mul_class_dir = ensemble_aln_msa(fasta_name, full_length_dir, serve_name, method = method)
	if DM_FLAG:
		real_dist_dm_dir_dict = {}
		mul_class_dm_dir_dict = {}
		for dm_fasta_name in dm_name_dict:
			domain_dir = outdir + '/%s/'%dm_fasta_name
			real_dist_dm_dir, mul_class_dm_dir = ensemble_aln_msa(dm_fasta_name, domain_dir, serve_name, method = method)
			real_dist_dm_dir_dict[dm_fasta_name] = real_dist_dm_dir
			mul_class_dm_dir_dict[dm_fasta_name] = mul_class_dm_dir
	# combine with domain
	if DM_FLAG:
		real_dist_dir, mul_class_dir = combine_dm_fl(full_length_dir, domain_dir_dict, dm_name_dict, serve_name, method = method)
	# "0 for dfold rr, 1 for dist rr, 2 for binary rr"
	if DF_FLAG == 0:
		real_dist_rr_folder, mul_class_rr_folder = generate_distrr_for_dfold(fasta_name, outdir, real_dist_dir, mul_class_dir)
		coneva_rd_rr_folder, coneva_mc_rr_folder = generate_distrr_for_coneva(fasta_name, outdir, real_dist_dir, mul_class_dir, method = method)
		src_file1 = real_dist_rr_folder + '/' + fasta_name + '.dist.rr'
		src_file2 = mul_class_rr_folder + '/' + fasta_name + '.dist.rr'
		src_file3 = real_dist_dir + '/real_dist/' + fasta_name + '.txt'
		src_file4 = coneva_rd_rr_folder + '/' + fasta_name + '.rr'
		dst_file1 = outdir + '/real_dist/'+fasta_name + '.dist.rr'
		dst_file2 = outdir + '/mul_class/'+fasta_name + '.dist.rr'
		dst_file3 = outdir + '/' + fasta_name + '.txt'
		dst_file4 = outdir + '/' + fasta_name + '.rr'
		chkdirs(dst_file1)
		chkdirs(dst_file2)
		chkdirs(dst_file3)
		chkdirs(dst_file4)
		if os.path.isfile(src_file1):
			shutil.copy(src_file1, dst_file1)
		if os.path.isfile(src_file2):
			shutil.copy(src_file2, dst_file2)
		if os.path.isfile(src_file3):
			shutil.copy(src_file3, dst_file3)
		if os.path.isfile(src_file4):
			shutil.copy(src_file4, dst_file4)
		print('Final %s real dist rr folder: %s'%(fasta_name, outdir + '/real_dist/'))
		print('Final %s mul class rr folder: %s'%(fasta_name, outdir + '/mul_class/'))
		if DM_FLAG:
			for dm_fasta_name in dm_name_dict:
				real_dist_dm_dir = real_dist_dm_dir_dict[dm_fasta_name]
				mul_class_dm_dir = mul_class_dm_dir_dict[dm_fasta_name]
				real_dist_dm_rr_folder, mul_class_dm_rr_folder = generate_distrr_for_dfold(dm_fasta_name, outdir, real_dist_dm_dir, mul_class_dm_dir)
				src_file1 = real_dist_dm_rr_folder + '/'+dm_fasta_name+'.dist.rr'
				src_file2 = mul_class_dm_rr_folder + '/'+dm_fasta_name+'.dist.rr'
				dst_file1 = outdir + '/real_dist/'+dm_fasta_name+'.dist.rr'
				dst_file2 = outdir + '/mul_class/'+dm_fasta_name+'.dist.rr'
				chkdirs(dst_file1)
				chkdirs(dst_file2)
				if os.path.isfile(src_file1):
					shutil.copy(src_file1, dst_file1)
				if os.path.isfile(src_file2):
					shutil.copy(src_file2, dst_file2)
				print('Final %s real dist rr folder: %s'%(dm_fasta_name, outdir + '/real_dist/'))
				print('Final %s mul class rr folder: %s'%(dm_fasta_name, outdir + '/mul_class/'))
	elif DF_FLAG == 1:
		mul_class_rr_folder = generate_distrr_for_dist(fasta_name, outdir, mul_class_dir, serve_name=serve_name)
		src_file1 = mul_class_rr_folder + '/'+fasta_name+'.rr'
		src_file2 = mul_class_dir + '/mul_class/' + fasta_name + '.npy'
		if serve_name == None:
			dst_file1 = outdir + '/distrib/' + fasta_name + '.rr'
			dst_file2 = outdir + '/distrib/' + fasta_name + '.npy'
			chkdirs(dst_file1)
		else:
			dst_file1 = outdir + '/distrib/' + serve_name + '/' + fasta_name + '.rr'
			dst_file2 = outdir + '/distrib/' + serve_name + '/' + fasta_name + '.npy'
			chkdirs(dst_file1)
		if os.path.isfile(src_file1):
			shutil.copy(src_file1, dst_file1)
			shutil.copy(src_file2, dst_file2)
		else:
			print("dist rr generate failed!\n")
		print('Final %s dist rr folder: %s'%(fasta_name, outdir + '/distrib/'))
		if DM_FLAG:
			for dm_fasta_name in dm_name_dict:
				mul_class_dm_dir = mul_class_dm_dir_dict[dm_fasta_name]
				src_file = mul_class_dm_dir + '/mul_class/' + dm_fasta_name + '.npy'
				if serve_name == None:
					dst_file = outdir + '/distrib/' + dm_fasta_name + '.npy'
					chkdirs(dst_file)
				else:
					dst_file = outdir + '/distrib/' + serve_name + '/' + dm_fasta_name + '.npy'
					chkdirs(dst_file)
				if os.path.isfile(src_file):
					shutil.copy(src_file, dst_file)

	elif DF_FLAG == 2:
		real_dist_rr_folder, mul_class_rr_folder = generate_distrr_for_contact(fasta_name, outdir, real_dist_dir, mul_class_dir, method = method, serve_name=serve_name)
		src_file1 = real_dist_rr_folder + '/' + fasta_name + '.rr'
		src_file2 = mul_class_rr_folder + '/' + fasta_name + '.rr'
		src_file3 = real_dist_dir + '/' + fasta_name + '.txt'
		src_file4 = mul_class_dir + '/bin_map_from_bin_avg/' + fasta_name + '.txt'
		if serve_name == None:
			dst_file1 = outdir + '/real_dist/' + fasta_name + '.rr'
			dst_file2 = outdir + '/mul_class/' + fasta_name + '.rr'
			dst_file3 = outdir + '/real_dist/' + fasta_name + '.txt'
			dst_file4 = outdir + '/mul_class/' + fasta_name + '.txt'
			chkdirs(dst_file1)
			chkdirs(dst_file2)
		else:
			dst_file1 = outdir + '/' + serve_name + '/real_dist/' + '/'  + fasta_name + '.rr'
			dst_file2 = outdir + '/' + serve_name + '/mul_class/' + '/'  + fasta_name + '.rr'
			dst_file3 = outdir + '/' + serve_name + '/real_dist/' + '/'  + fasta_name + '.txt'
			dst_file4 = outdir + '/' + serve_name + '/mul_class/' + '/'  + fasta_name + '.txt'
			chkdirs(dst_file1)
			chkdirs(dst_file2)
		if os.path.isfile(src_file1):
			shutil.copy(src_file1, dst_file1)
		if os.path.isfile(src_file2):
			shutil.copy(src_file2, dst_file2)
		if os.path.isfile(src_file3):
			shutil.copy(src_file3, dst_file3)
		if os.path.isfile(src_file4):
			shutil.copy(src_file4, dst_file4)
		print('Final %s real dist rr folder: %s'%(fasta_name, outdir + '/real_dist/'))
		print('Final %s mul class rr folder: %s'%(fasta_name, outdir + '/mul_class/'))

	sys.exit(outdir)
