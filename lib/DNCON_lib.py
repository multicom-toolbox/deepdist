# -*- coding: utf-8 -*-


from shutil import copyfile
import platform
import os
import numpy as np
import math
import sys
import random
import keras.backend as K
import itertools
from operator import itemgetter
from sklearn.metrics import recall_score, f1_score, confusion_matrix, matthews_corrcoef, precision_score

epsilon = K.epsilon()
# from Data_loading import getX_1D_2D,getX_2D_format

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def chkfiles(fn):
  if os.path.exists(fn):
    return True 
  else:
    return False

def build_dataset_dictionaries(path_lists):
  length_dict = {}
  n_dict = {}
  neff_dict = {}
  with open(path_lists + 'L.txt') as f:
    for line in f:
      cols = line.strip().split()
      length_dict[cols[0]] = int(cols[1])
  with open(path_lists + 'N.txt') as f:
    for line in f:
      cols = line.strip().split()
      n_dict[cols[0]] = int(float(cols[1]))
  with open(path_lists + 'Neff.txt') as f:
    for line in f:
      cols = line.strip().split()
      neff_dict[cols[0]] = int(float(cols[1]))
  tr_l = {}
  tr_n = {}
  tr_e = {}
  with open(path_lists + 'train.lst') as f:
    for line in f:
      tr_l[line.strip()] = length_dict[line.strip()]
      tr_n[line.strip()] = n_dict[line.strip()]
      tr_e[line.strip()] = neff_dict[line.strip()]
  te_l = {}
  te_n = {}
  te_e = {}
  with open(path_lists + 'test.lst') as f:
    for line in f:
      te_l[line.strip()] = length_dict[line.strip()]
      te_n[line.strip()] = n_dict[line.strip()]
      te_e[line.strip()] = neff_dict[line.strip()]
  print ('')
  print ('Data counts:')
  print ('Total : ' + str(len(length_dict)))
  print ('Train : ' + str(len(tr_l)))
  print ('Test  : ' + str(len(te_l)))
  print ('')
  return (tr_l, tr_n, tr_e, te_l, te_n, te_e)


def build_dataset_dictionaries_other(path_lists, list_name):
  length_dict = {}
  with open(path_lists + 'L.txt') as f:
    for line in f:
      cols = line.strip().split()
      length_dict[cols[0]] = int(cols[1])
  tr_l = {}
  with open(path_lists + list_name) as f:
    for line in f:
      if line.strip() not in length_dict:
        continue
      else:
        tr_l[line.strip()] = length_dict[line.strip()]
  # print ('Data counts:')
  # print ('Total : ' + str(len(length_dict)))
  # print ('Train : ' + str(len(tr_l)))
  return (tr_l)

def build_dataset_dictionaries_train(path_lists):
  length_dict = {}
  with open(path_lists + 'L.txt') as f:
    for line in f:
      cols = line.strip().split()
      length_dict[cols[0]] = int(cols[1])
  tr_l = {}
  with open(path_lists + 'train.lst') as f:
    for line in f:
      if line.strip() not in length_dict:
        continue
      else:
        tr_l[line.strip()] = length_dict[line.strip()]
  return (tr_l)

def build_dataset_dictionaries_test(path_lists):
  length_dict = {}
  with open(path_lists + 'L.txt') as f:
    for line in f:
      cols = line.strip().split()
      length_dict[cols[0]] = int(cols[1])
  te_l = {}
  with open(path_lists + 'test.lst') as f:
    for line in f:
      if line.strip() not in length_dict:
        continue
      else:
        te_l[line.strip()] = length_dict[line.strip()]
  return (te_l)

def build_dataset_dictionaries_sample(path_lists):
  length_dict = {}
  with open(path_lists + 'L.txt') as f:
    for line in f:
      cols = line.strip().split()
      length_dict[cols[0]] = int(cols[1])
  ex_l = {}
  with open(path_lists + 'sample.lst') as f:
    for line in f:
      if line.strip() not in length_dict:
        continue
      else:
        ex_l[line.strip()] = length_dict[line.strip()]
  return (ex_l)

def subset_pdb_dict(dict, minL, maxL, count, randomize_flag):
  selected = {}
  # return a dict with random 'X' PDBs
  if (randomize_flag == 'random'):
    pdbs = list(dict.keys())
    sys.stdout.flush()
    random.shuffle(pdbs)
    i = 0
    for pdb in pdbs:
      if (dict[pdb] > minL and dict[pdb] <= maxL):
        selected[pdb] = dict[pdb]
        i = i + 1
        if i == count:
          break
  # return first 'X' PDBs sorted by L
  if (randomize_flag == 'ordered'):
    i = 0
    for key, value in sorted(dict.items(), key=lambda  x: x[1]):
      if (dict[key] > minL and dict[key] <= maxL):
        selected[key] = value
        i = i + 1
        if i == count:
          break
  return selected

def load_sample_data_2D(data_list, path_of_X, path_of_Y,seq_end, min_seq_sep,dist_string, reject_fea_file='None'):
  import pickle
  data_all_dict = dict()
  print("######### Loading data\n")
  accept_list = []
  notxt_flag = True
  if reject_fea_file != 'None':
    with open(reject_fea_file) as f:
      for line in f:
        if line.startswith('#'):
          feature_name = line.strip()
          feature_name = feature_name[0:]
          accept_list.append(feature_name)
  ex_l = build_dataset_dictionaries_sample(data_list)
  sample_dict = subset_pdb_dict(ex_l, 0, 500, seq_end, 'random') #can be random ordered
  sample_name = list(sample_dict.keys())
  sample_lens = list(sample_dict.values())
  feature_num = 0
  for i in range(0,len(sample_name)):
    pdb_name = sample_name[i]
    pdb_lens = sample_lens[i]
    print(pdb_name, "..",end='')
    
    featurefile =path_of_X + '/other/' + 'X-'  + pdb_name + '.txt'
    if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list and '# plmc' not in accept_list and '# pre' not in accept_list and '# dca' not in accept_list and '# err' not in accept_list and '# netout' not in accept_list)) or 
          (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list or '# plmc' not in accept_list or '# pre' not in accept_list or '# dca' not in accept_list or '# err' not in accept_list or '# netout' not in accept_list)) or (len(accept_list) > 2)):
      notxt_flag = False
      if not os.path.isfile(featurefile):
                  print("feature file not exists: ",featurefile, " pass!")
                  continue     
    cov = path_of_X + '/cov/' + pdb_name + '.cov'
    if '# cov' in accept_list:
      if not os.path.isfile(cov):
                  print("Cov Matrix file not exists: ",cov, " pass!")
                  continue        
    plm = path_of_X + '/plm/' + pdb_name + '.plm'
    if '# plm' in accept_list:
      if not os.path.isfile(plm):
                  print("plm matrix file not exists: ",plm, " pass!")
                  continue     
    plmc = path_of_X + '/plmc/' + pdb_name + '.plmc'
    if '# plmc' in accept_list:
      if not os.path.isfile(plmc):
                  print("plmc matrix file not exists: ",plmc, " pass!")
                  continue            
    pre = path_of_X + '/pre/' + pdb_name + '.pre'
    if '# pre' in accept_list:
      if not os.path.isfile(pre):
                  print("pre matrix file not exists: ",pre, " pass!")
                  continue 
    dca = path_of_X + '/dca/' + pdb_name + '.dca'
    if '# dca' in accept_list:
      if not os.path.isfile(dca):
                  print("dca matrix file not exists: ",dca, " pass!")
                  continue 
    err = path_of_X + '/dist_error/' + pdb_name + '.txt'
    if '# err' in accept_list:
      if not os.path.isfile(err):
                  print("err matrix file not exists: ",err, " pass!")
                  continue 
    netout = path_of_X + '/net_out/' + pdb_name + '.npy'
    if '# netout' in accept_list:      
      if not os.path.isfile(netout):
                  print("netout matrix file not exists: ",netout, " pass!")
                  continue  
    aa = path_of_X + '/aa/' + pdb_name + '.aa'
    if '# aa' in accept_list:      
      if not os.path.isfile(aa):
                  print("netout matrix file not exists: ",aa, " pass!")
                  continue  

    ### load the data
    (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, plmc, pre, dca, err, netout, aa, accept_list, pdb_lens, notxt_flag)
    # print("\n######",len(featuredata))

    feature_num = len(featuredata)
  return feature_num

def get_all_file_contain_str(dir, list):
    files = os.listdir(dir)
    tar_filename = []
    for file in files:
        if list in file:
            tar_filename.append(file)

    return tar_filename

def get_sub_map_index(index_file):
  index_list = np.loadtxt(index_file)[:,1]
  sub_map_index = [index_list[0] - 1, index_list[-1] - 1]
  sub_map_index = list(map(int, sub_map_index))
  summary=[]
  for k, g in itertools.groupby(enumerate(index_list), lambda x: x[1]-x[0]):
      summary.append(list(map(itemgetter(1), g)))
  sub_map_gap = []
  for i in range(len(summary)-1):
      # list file start from 1, array start from 0
      left = int(summary[i][-1] + 1) - 1
      right = int(summary[i+1][0] -1) - 1
      sub_map_gap.append([left, right])
  return sub_map_index, sub_map_gap

def get_y_from_this_list_casp(selected_ids, path, index_path, min_seq_sep, l_max, y_dist, lable_type = 'bin'):
  sample_pdb = ''
  for pdb in selected_ids:
    file_names = get_all_file_contain_str(path, pdb)
    # print(file_names)
    Y = np.zeros((len(file_names), l_max * l_max))
    sub_map_index = list()
    sub_map_gap = list()
    if len(file_names) > 1:
      for i in range(len(file_names)):
          sub_map_index1, sub_map_gap1 = get_sub_map_index(index_path+'/'+file_names[i].split('-')[1]+'-'+file_names[i].split('-')[2])
          sub_map_index.append(sub_map_index1)
          sub_map_gap.append(sub_map_gap1)
          Y[i,:] =  getY(path + '/' + file_names[i], min_seq_sep, l_max)
    else:
      sub_map_index1, sub_map_gap1 = get_sub_map_index(index_path+'/'+file_names[0].split('-')[1]+'-'+file_names[0].split('-')[2])
      sub_map_index.append(sub_map_index1)
      sub_map_gap.append(sub_map_gap1)
      Y[0,:] =  getY(path + '/' + file_names[0], min_seq_sep, l_max)
  return Y, sub_map_index, sub_map_gap


def get_y_from_this_list(selected_ids, path, min_seq_sep, l_max, y_dist, lable_type = 'bin'):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  if lable_type == 'bin':
    y_file = path + 'Y' + y_dist + '-' + sample_pdb + '.txt'
    if not os.path.isfile(y_file):
      print("%s not exits!" % y_file)
      return False
    y = getY(y_file, min_seq_sep, l_max)
  elif lable_type == 'real':
    y_file = path + sample_pdb + '.txt'
    if not os.path.isfile(y_file):
      print("%s not exits!" % y_file)
      return False
    y = getY(y_file, 0, l_max)
  if (l_max * l_max != len(y)):
    print ('Error!! y does not have L * L feature values!!')
    return False
  Y = np.zeros((xcount, l_max * l_max))
  i = 0
  if lable_type == 'bin':
    for pdb in sorted(selected_ids):
      Y[i, :]  = getY(path + 'Y' + y_dist + '-' + pdb + '.txt', min_seq_sep, l_max)
      i = i + 1
  elif lable_type == 'real':
    for pdb in sorted(selected_ids):
      Y[i, :]  = getY(path + pdb + '.txt', 0, l_max)
      i = i + 1
  return Y

def get_y_from_this_list_dist_error(selected_ids, path, min_seq_sep, l_max):
  Y = []
  for pdb in sorted(selected_ids):
    y1  = getY(path[0] + '/' + pdb + '.txt', min_seq_sep, l_max)
    y2  = getY(path[1] + '/' + pdb + '.error', min_seq_sep, l_max)
    y = np.concatenate((y1.reshape(l_max, l_max, 1), y2.reshape(l_max, l_max, 1)), axis=-1)
    Y.append(y)
  Y = np.array(Y)
  return Y

#get binary y lable
def getY(true_file, min_seq_sep, l_max):
  # calcualte the length of the protein (the first feature)
  L = 0
  with open(true_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      L = line.strip().split()
      L = len(L)
      break
  Y = np.zeros((l_max, l_max))
  i = 0
  if L > l_max:
    return Y
  with open(true_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      this_line = line.strip().split()
      if len(this_line) != L:
        print("\nThis_line = %i, L = %i, Lable file %s error!\n"%(len(this_line), L, true_file))
        Y = [0]
        return  Y
      Y[i, 0:L] = np.asarray(this_line)

      i = i + 1
  for p in range(0,L):
    for q in range(0,L):
      # updated only for the last project 'p19' to test the effect
      if ( abs(q - p) < min_seq_sep):
        Y[p][q] = 0
  Y = Y.flatten()
  return Y
 
def getY_4(true_file, min_seq_sep, l_max):
  # calcualte the length of the protein (the first feature)
  Y = np.zeros((l_max, l_max))
  y = np.loadtxt(true_file)
  L = y.shape[0]
  y_sep = np.triu(y, min_seq_sep) + np.tril(y, -min_seq_sep)
  Y[0:L,0:L] = y_sep
  Y = Y.flatten()
  return Y

def get_x_from_this_list(selected_ids, path, l_max):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  print(path,'/X-',sample_pdb,'.txt')
  x = getX(path + 'X-'  + sample_pdb + '.txt', l_max)
  F = len(x[0, 0, :])
  X = np.zeros((xcount, l_max, l_max, F))
  i = 0
  for pdb in sorted(selected_ids):
    T = getX(path + 'X-'  + pdb + '.txt', l_max)
    if len(T[0, 0, :]) != F:
      print('ERROR! Feature length of ',sample_pdb,' not equal to ',pdb)
    X[i, :, :, :] = T
    i = i + 1
  return X

def getX_1D_2D(feature_file, cov, plm, reject_fea_file='None'):
  # calcualte the length of the protein (the first feature)
  reject_list = []
  reject_list.append('# PSSM')
  reject_list.append('# AA composition')
  #print("Checking ",reject_fea_file)
  if reject_fea_file != 'None':
    #print("Loading ",reject_fea_file)
    with open(reject_fea_file) as f:
      for line in f:
        if line.startswith('-'):
          feature_name = line.strip()
          feature_name = feature_name[1:]
          #print("Removing ",feature_name)
          reject_list.append(feature_name)
  L = 0
  with open(feature_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      L = line.strip().split()
      L = int(round(math.exp(float(L[0]))))
      break
  Data = []
  feature_all_dict = dict()
  feature_index_all_dict = dict() # to make sure the feature are same ordered 
  feature_name='None'
  feature_index=0;
  with open(feature_file) as f:
    accept_flag = 1
    for line in f:
      if line.startswith('#'):
        if line.strip() in reject_list:
          accept_flag = 0
        else:
          accept_flag = 1
        feature_name = line.strip()
        continue
      if accept_flag == 0:
        continue
      
      if line.startswith('#'):
        continue
      this_line = line.strip().split()
      if len(this_line) == 0:
        continue
      if len(this_line) == 1:
        # 0D feature
        feature_namenew = feature_name + ' 0D'
        feature_index +=1
        if feature_index in feature_index_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_index_all_dict[feature_index] = feature_namenew
        
        feature0D = np.zeros((1, L))
        feature0D[0, :] = float(this_line[0])
        if feature_index in feature_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_all_dict[feature_index] = feature0D
      elif len(this_line) == L:
        # 1D feature
        feature1D = np.zeros((1, L))
        feature_namenew = feature_name + ' 1D'
        feature_index +=1
        if feature_index in feature_index_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_index_all_dict[feature_index] = feature_namenew
        
        for i in range (0, L):
          feature1D[0, i] = float(this_line[i])
        if feature_index in feature_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_all_dict[feature_index] = feature1D
      elif len(this_line) == L * L:
        # 2D feature
        feature2D = np.asarray(this_line).reshape(L, L)
        feature_namenew = feature_name + ' 2D'
        feature_index +=1
        if feature_index in feature_index_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_index_all_dict[feature_index] = feature_namenew
        if feature_index in feature_all_dict:
          print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          exit;
        else:
          feature_all_dict[feature_index] = feature2D
      else:
        print (line)
        print ('Error!! Unknown length of feature in !!' + feature_file)
        print ('Expected length 0, ' + str(L) + ', or ' + str (L*L) + ' - Found ' + str(len(this_line)))
        sys.exit()
  if '# cov' not in reject_list:
      cov_rawdata = np.fromfile(cov, dtype=np.float32)
      length = int(math.sqrt(cov_rawdata.shape[0]/21/21))
      if length != L:
          print("Bad Alignment, pls check!")
          exit;
      inputs_cov = cov_rawdata.reshape(1,441,L,L)
      for i in range(441):
          feature2D = inputs_cov[0][i]
          feature_namenew = '# Covariance Matrix '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              exit;
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              exit;
          else:
              feature_all_dict[feature_index] = feature2D
  if '# plm' not in reject_list:
    plm_rawdata = np.fromfile(plm, dtype=np.float32)
    length = int(math.sqrt(plm_rawdata.shape[0]/21/21))
    if length != L:
        print("Bad Alignment, pls check!")
        exit;
    inputs_plm = plm_rawdata.reshape(1,441,L,L)
    for i in range(441):
        feature2D = inputs_plm[0][i]
        feature_namenew = '# Pseudo_Likelihood Maximization '+str(i+1)+ ' 2D'
        feature_index +=1
        if feature_index in feature_index_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit;
        else:
            feature_index_all_dict[feature_index] = feature_namenew
        if feature_index in feature_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit;
        else:
            feature_all_dict[feature_index] = feature2D
  return (feature_all_dict,feature_index_all_dict)

def get_x_1D_2D_from_this_list(selected_ids, feature_dir, l_max,dist_string, reject_fea_file='None'):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  featurefile = feature_dir + 'X-'  + sample_pdb + '.txt'
  cov = feature_dir + '/'  + sample_pdb + '.cov'
  plm = feature_dir + '/'  + sample_pdb + '.plm'
  #print(featurefile)
  ### load the data
  (featuredata,feature_index_all_dict) = getX_1D_2D(featurefile, cov, plm, reject_fea_file=reject_fea_file)     
  ### merge 1D data to L*m
  ### merge 2D data to  L*L*n
  feature_1D_all=[]
  feature_2D_all=[]
  for key in sorted(featuredata.keys()):
      featurename = feature_index_all_dict[key]
      feature = featuredata[key]
      feature = np.asarray(feature)
      #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
      #print "keys: ", key, ": ", featuredata[key].shape
      
      if feature.shape[0] == feature.shape[1]:
        feature_2D_all.append(feature)
      else:
        feature_1D_all.append(feature)
  
  fea_len = feature_2D_all[0].shape[0]
  F_2D = len(feature_2D_all)
  
  X_2D_tmp = np.zeros((fea_len, fea_len, F_2D))
  for m in range (0, F_2D):
    X_2D_tmp[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
    
  
  F_1D = len(feature_1D_all)
  
  X_1D_tmp = np.zeros((fea_len, F_1D))
  for m in range (0, F_1D):
    X_1D_tmp[0:fea_len, m] = feature_1D_all[m]
  
  
  feature_1D_all_complete =  X_1D_tmp
  #feature_1D_all_complete.shape #(123, 22)
  feature_2D_all_complete =  X_2D_tmp
  #feature_2D_all_complete.shape #(123, 123, 18)
  fea_len = feature_2D_all_complete.shape[0]
  F_1D = len(feature_1D_all_complete[0, :])
  F_2D = len(feature_2D_all_complete[0, 0, :])
  X_1D = np.zeros((xcount, l_max, F_1D))
  X_2D = np.zeros((xcount, l_max, l_max, F_2D))
  pdb_indx = 0
  for pdb_name in sorted(selected_ids):
      print(pdb_name, "..",end='')
      featurefile = feature_dir + '/X-' + pdb_name + '.txt'
      cov = feature_dir + '/' + pdb_name + '.cov'
      if not os.path.isfile(featurefile):
                  print("feature file not exists: ",featurefile, " pass!")
                  continue        
      plm = feature_dir + '/' + pdb_name + '.plm'
      if not os.path.isfile(plm):
                  print("plm matrix file not exists: ",plm, " pass!")
                  continue          
      targetfile = feature_dir + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
      if not os.path.isfile(targetfile):
                  print("target file not exists: ",targetfile, " pass!")
                  continue
      ### load the data
      (featuredata,feature_index_all_dict) = getX_1D_2D(featurefile, cov, plm, reject_fea_file=reject_fea_file)     
      ### merge 1D data to L*m
      ### merge 2D data to  L*L*n
      feature_1D_all=[]
      feature_2D_all=[]
      for key in sorted(featuredata.keys()):
          featurename = feature_index_all_dict[key]
          feature = featuredata[key]
          feature = np.asarray(feature)
          #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
          #print "keys: ", key, ": ", featuredata[key].shape
          
          if feature.shape[0] == feature.shape[1]:
            feature_2D_all.append(feature)
          else:
            feature_1D_all.append(feature)
      
      fea_len = feature_2D_all[0].shape[0]
      F_2D = len(feature_2D_all)
      
      X_2D_tmp = np.zeros((fea_len, fea_len, F_2D))
      for m in range (0, F_2D):
        X_2D_tmp[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
        
      
      F_1D = len(feature_1D_all)
      
      X_1D_tmp = np.zeros((fea_len, F_1D))
      for m in range (0, F_1D):
        X_1D_tmp[0:fea_len, m] = feature_1D_all[m]
      
      
      feature_1D_all_complete =  X_1D_tmp
      #feature_1D_all_complete.shape #(123, 22)
      feature_2D_all_complete =  X_2D_tmp
      #feature_2D_all_complete.shape #(123, 123, 18)
      if len(feature_1D_all_complete[0, :]) != F_1D:
        print('ERROR! 1D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        exit;

      ### expand to lmax
      if feature_1D_all_complete.shape[0] < l_max:
        L = feature_1D_all_complete.shape[0]
        F = feature_1D_all_complete.shape[1]
        X_tmp = np.zeros((l_max, F))
        for i in range (0, F):
          X_tmp[0:L, i] = feature_1D_all_complete[:,i]
        feature_1D_all_complete = X_tmp
      X_1D[pdb_indx, :, :] = feature_1D_all_complete
      if len(feature_2D_all_complete[0, 0, :]) != F_2D:
        print('ERROR! 2D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        exit;

      ### expand to lmax
      if feature_2D_all_complete.shape[0] < l_max:
        L = feature_2D_all_complete.shape[0]
        F = feature_2D_all_complete.shape[2]
        X_tmp = np.zeros((l_max, l_max, F))
        for i in range (0, F):
          X_tmp[0:L,0:L, i] = feature_2D_all_complete[:,:,i]
        feature_2D_all_complete = X_tmp
      X_2D[pdb_indx, :, :, :] = feature_2D_all_complete
      pdb_indx = pdb_indx + 1
  return (X_1D,X_2D)

def getX_2D_format(feature_file, cov, plm, plmc, pre, dca, err, netout, aa, accept_list, pdb_len = 0, notxt_flag = True, logfile = None):
  # calcualte the length of the protein (the first feature)
  if logfile != None:
    chkdirs(logfile)

  L = 0
  Data = []
  feature_all_dict = dict()
  feature_index_all_dict = dict() # to make sure the feature are same ordered 
  feature_name='None'
  feature_index=0
  # print(reject_list)
  if notxt_flag == True:
    L = pdb_len
  else:
    with open(feature_file) as f:
      for line in f:
        if line.startswith('#'):
          continue
        L = line.strip().split()
        L = int(round(math.exp(float(L[0]))))
        break
    with open(feature_file) as f:
      accept_flag = 1
      for line in f:
        if line.startswith('#'):
          if line.strip() not in accept_list:
            accept_flag = 0
          else:
            accept_flag = 1
          feature_name = line.strip()
          continue
        if accept_flag == 0:
          continue
        
        if line.startswith('#'):
          continue
        this_line = line.strip().split()
        if len(this_line) == 0:
          continue
        if len(this_line) == 1:
          # 0D feature
          continue
          # feature_namenew = feature_name + ' 0D'
          # feature_index +=1
          # if feature_index in feature_index_all_dict:
          #   print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          #   exit;
          # else:
          #   feature_index_all_dict[feature_index] = feature_namenew

          # feature0D = np.zeros((L, L))
          # feature0D[:, :] = float(this_line[0])
          # #feature0D = np.zeros((1, L))
          # #feature0D[0, :] = float(this_line[0])
          
          # if feature_index in feature_all_dict:
          #   print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
          #   exit;
          # else:
          #   feature_all_dict[feature_index] = feature0D
        elif len(this_line) == L:
          # 1D feature
          # continue
          feature1D1 = np.zeros((L, L))
          feature1D2 = np.zeros((L, L))
          for i in range (0, L):
            feature1D1[i, :] = float(this_line[i])
            feature1D2[:, i] = float(this_line[i])
          
          ### load feature 1
          feature_index +=1
          feature_namenew = feature_name + ' 1D1'
          if feature_index in feature_index_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit;
          else:
            feature_index_all_dict[feature_index] = feature_namenew
          
          if feature_index in feature_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit;
          else:
            feature_all_dict[feature_index] = feature1D1
          
          ### load feature 2
          feature_index +=1
          feature_namenew = feature_name + ' 1D2'
          if feature_index in feature_index_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit;
          else:
            feature_index_all_dict[feature_index] = feature_namenew

          if feature_index in feature_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit;
          else:
            feature_all_dict[feature_index] = feature1D2
        elif len(this_line) == L * L:
          # 2D feature
          feature2D = np.asarray(this_line).reshape(L, L)
          feature_index +=1
          feature_namenew = feature_name + ' 2D'
          if feature_index in feature_index_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit
          else:
            feature_index_all_dict[feature_index] = feature_namenew
          
          if feature_index in feature_all_dict:
            print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
            exit
          else:
            feature_all_dict[feature_index] = feature2D
        else:
          # print (line)
          if logfile != None:
            with open(logfile, "a") as myfile:
              myfile.write('Error!! Unknown length of feature in !!' + feature_file)
              myfile.write('Expected length 0, ' + str(L) + ', or ' + str (L*L) + ' - Found ' + str(len(this_line)))
            return False, False
          else:
            print ('Error!! Unknown length of feature in !!' + feature_file)
            print ('Expected length 0, ' + str(L) + ', or ' + str (L*L) + ' - Found ' + str(len(this_line)))
            return False, False
  #Add Covariance Matrix 
  if '# cov' in accept_list:   
      cov_rawdata = np.fromfile(cov, dtype=np.float32)
      length = int(math.sqrt(cov_rawdata.shape[0]/21/21))
      if length != L:
          print("Cov Bad Alignment, want %d get %d, pls check! %s" %(L, length, cov))
          if logfile != None:
            with open(logfile, "a") as myfile:
              myfile.write("Cov Bad Alignment, pls check! %s\n" %(cov))
            return False, False
          else:
            return False, False
            # sys.exit()
      inputs_cov = cov_rawdata.reshape(1,441,L,L) #????
      for i in range(441):
          feature2D = inputs_cov[0][i]
          feature_namenew = '# Covariance Matrix '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_all_dict[feature_index] = feature2D
  #Add Pseudo_Likelihood Maximization
  if '# plm' in accept_list:  
      plm_rawdata = np.fromfile(plm, dtype=np.float32)
      length = int(math.sqrt(plm_rawdata.shape[0]/21/21))
      if length != L:
          print("Plm Bad Alignment, want %d get %d, pls check! %s" %(L, length, plm))
          if logfile != None:
            with open(logfile, "a") as myfile:
              myfile.write("Plm Bad Alignment, pls check! %s\n" %(plm))
            return False, False
          else:
            return False, False
            # sys.exit()
      inputs_plm = plm_rawdata.reshape(1,441,L,L)
      for i in range(441):
          feature2D = inputs_plm[0][i]
          feature_namenew = '# Pseudo_Likelihood Maximization '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_all_dict[feature_index] = feature2D
  #Add Pseudo_Likelihood Maximization
  if '# plmc' in accept_list:  
      plmc_rawdata = np.fromfile(plmc, dtype=np.float32)
      length = int(math.sqrt(plmc_rawdata.shape[0]/21/21))
      if length != L:
          print("plmc Bad Alignment, want %d get %d, pls check! %s" %(L, length, plmc))
          if logfile != None:
            with open(logfile, "a") as myfile:
              myfile.write("plmc Bad Alignment, pls check! %s\n" %(plmc))
            return False, False
          else:
            return False, False
            # sys.exit()
      inputs_plmc = plmc_rawdata.reshape(1,441,L,L)
      for i in range(441):
          feature2D = inputs_plmc[0][i]
          feature_namenew = '# Pseudo_Likelihood Maximization '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_all_dict[feature_index] = feature2D
  if '# pre' in accept_list:  
      pre_rawdata = np.fromfile(pre, dtype=np.float32)
      length = int(math.sqrt(pre_rawdata.shape[0]/21/21))
      if length != L:
          print("Pre Bad Alignment, want %d get %d, pls check! %s" %(L, length, pre))
          if logfile != None:
            with open(logfile, "a") as myfile:
              myfile.write("Pre Bad Alignment, pls check! %s\n" %(pre))
            return False, False
          else:
            return False, False
            # sys.exit()
      inputs_pre = pre_rawdata.reshape(1,441,L,L)
      for i in range(441):
          feature2D = inputs_pre[0][i]
          feature_namenew = '# Pre Maximization '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_all_dict[feature_index] = feature2D
  if '# dca' in accept_list:  
      dca_rawdata = np.load(dca)
      length = dca_rawdata.shape[1]
      if length != L or np.isnan(np.min(dca_rawdata)):
          print("dca Bad Alignment, want %d get %d, pls check! %s" %(L, length, dca))
          if logfile != None:
            with open(logfile, "a") as myfile:
              myfile.write("dca Bad Alignment, pls check! %s\n" %(dca))
            return False, False
          else:
            return False, False
            # sys.exit()
      # inputs_dca = dca_rawdata.reshape(1,526,length,length)
      inputs_dca =  dca_rawdata.transpose(0, 3, 1, 2)
      for i in range(526):
          feature2D = inputs_dca[0][i]
          feature_namenew = '# dca Maximization '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_all_dict[feature_index] = feature2D
  if '# err' in accept_list:
      err_raw=np.loadtxt(err)
      length = err_raw.shape[0]
      chn = 1
      if length != L:
          print("Err Bad Feature, pls check!")
          if logfile != None:
            with open(logfile, "a") as myfile:
              myfile.write("Err Bad Feature, pls check! %s\n" %(err))
            return False, False
          else:
            return False, False
      inputs_err =  err_raw.reshape(1,chn,L,L)
      for i in range(chn):
          feature2D = inputs_err[0][i]
          feature_namenew = '# Dist Error '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_all_dict[feature_index] = feature2D
  if '# netout' in accept_list:
      netout_raw=np.load(netout)
      length = netout_raw.shape[0]
      chn = netout_raw.shape[-1]
      if length != L:
          print("Net Bad Alignment, pls check!")
          return False, False
          # sys.exit()
      inputs_netout =  netout_raw.transpose(2, 0, 1)
      inputs_netout =  inputs_netout.reshape(1,chn,L,L)
      for i in range(chn):
          feature2D = inputs_netout[0][i]
          feature_namenew = '# Network output '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_all_dict[feature_index] = feature2D
  if '# aa' in accept_list:
      aa_raw = np.fromfile(aa, dtype=np.uint8)
      length = int(np.sqrt(aa_raw.shape[0]/42))
      chn = 42
      if length != L:
          print("Net Bad Alignment, pls check!")
          return False, False
          # sys.exit()
      inputs_aa = np.reshape(aa_raw, (length, length, chn))
      inputs_aa = inputs_aa.transpose(2,0,1)
      inputs_aa = inputs_aa.reshape(1,chn,length,length)
      for i in range(chn):
          feature2D = inputs_aa[0][i]
          feature_namenew = '# Network output '+str(i+1)+ ' 2D'
          feature_index +=1
          if feature_index in feature_index_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_index_all_dict[feature_index] = feature_namenew
          if feature_index in feature_all_dict:
              print("Duplicate feature name ",feature_namenew, " in file ",feature_file)
              sys.exit()
          else:
              feature_all_dict[feature_index] = feature2D

  return (feature_all_dict,feature_index_all_dict)

def getX_2D_format_4(feature_file, cov, plm, pre, netout, accept_list, pdb_len = 0, notxt_flag = True):
  # calcualte the length of the protein (the first feature)
  L = 0
  Data = []
  feature_all_list = []
  feature_index_all_dict = dict() # to make sure the feature are same ordered 
  feature_name='None'
  feature_index=0
  # print(reject_list)
  if notxt_flag == True:
    L = pdb_len
  else:
    with open(feature_file) as f:
      for line in f:
        if line.startswith('#'):
          continue
        L = line.strip().split()
        L = int(round(math.exp(float(L[0]))))
        break
    with open(feature_file) as f:
      accept_flag = 1
      for line in f:
        if line.startswith('#'):
          if line.strip() not in accept_list:
            accept_flag = 0
          else:
            accept_flag = 1
          feature_name = line.strip()
          continue
        if accept_flag == 0:
          continue
        if line.startswith('#'):
          continue
        this_line = line.strip().split()
        if len(this_line) == 0:
          continue
        if len(this_line) == 1:
          # 0D feature
          continue
        elif len(this_line) == L:
          # 1D feature
          continue
        elif len(this_line) == L * L:
          # 2D feature
          feature2D = np.asarray(this_line).reshape(L, L, 1)
          if len(feature_all_list) < 1:
            feature_all_list = feature2D
          else:
            feature_all_list = np.append(feature_all_list, feature2D, axis = -1)
  #Add Covariance Matrix 
  if '# cov' in accept_list:   
      cov_rawdata = np.fromfile(cov, dtype=np.float32)
      length = int(math.sqrt(cov_rawdata.shape[0]/21/21))
      if length != L:
          print("Cov Bad Alignment, pls check! %s" %(cov))
          sys.exit()
      inputs_cov = cov_rawdata.reshape(441,L,L) #????
      inputs_cov = inputs_cov.transpose(1, 2, 0)
      if len(feature_all_list) < 1:
        feature_all_list = inputs_cov
      else:
        feature_all_list = np.append(feature_all_list, inputs_cov, axis = -1)
  #Add Pseudo_Likelihood Maximization
  if '# plm' in accept_list:  
      plm_rawdata = np.fromfile(plm, dtype=np.float32)
      length = int(math.sqrt(plm_rawdata.shape[0]/21/21))
      if length != L:
          print("Plm Bad Alignment, pls check! %s" %(plm))
          sys.exit()
      inputs_plm = plm_rawdata.reshape(441,L,L)
      inputs_plm = inputs_plm.transpose(1, 2, 0)
      if len(feature_all_list) < 1:
        feature_all_list = inputs_plm
      else:
        feature_all_list = np.append(feature_all_list, inputs_plm, axis = -1)

  if '# pre' in accept_list:  
      pre_rawdata = np.fromfile(pre, dtype=np.float32)
      length = int(math.sqrt(pre_rawdata.shape[0]/21/21))
      if length != L:
          print("Pre Bad Alignment, pls check! %s" %(plm))
          sys.exit()
      inputs_pre = pre_rawdata.reshape(441,L,L)
      inputs_pre = inputs_pre.transpose(1, 2, 0)
      if len(feature_all_list) < 1:
        feature_all_list = inputs_pre
      else:
        feature_all_list = np.append(feature_all_list, inputs_pre, axis = -1)

  if '# netout' in accept_list:
      netout_raw=np.load(netout)
      length = netout_raw.shape[0]
      chn = netout_raw.shape[-1]
      if length != L:
          print("Net Bad Alignment, pls check!")
          sys.exit()

      if len(feature_all_list) < 1:
        feature_all_list = netout_raw
      else:
        feature_all_list = np.append(feature_all_list, netout_raw, axis = -1)

  return feature_all_list

def get_x_2D_from_this_list(selected_ids, feature_dir, l_max,dist_string, reject_fea_file='None', pdb_len = 0):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  accept_list = []
  notxt_flag = True
  if reject_fea_file != 'None':
    with open(reject_fea_file) as f:
      for line in f:
        if line.startswith('#'):
          feature_name = line.strip()
          feature_name = feature_name[0:]
          accept_list.append(feature_name)

  featurefile =feature_dir + '/other/' + 'X-'  + sample_pdb + '.txt'
  cov =feature_dir + '/cov/'  + sample_pdb + '.cov'
  plm =feature_dir + '/plm/'  + sample_pdb + '.plm'
  plmc =feature_dir + '/plmc/'  + sample_pdb + '.plmc'
  pre =feature_dir + '/pre/'  + sample_pdb + '.pre'
  dca =feature_dir + '/dca/'  + sample_pdb + '.dca'
  err =feature_dir + '/dist_error/'  + sample_pdb + '.txt'
  netout = feature_dir + '/net_out/' + sample_pdb + '.npy'
  aa = feature_dir + '/aa/' + sample_pdb + '.aa'
  # print(featurefile)
  if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list and '# plmc' not in accept_list and '# pre' not in accept_list and '# dca' not in accept_list and '# err' not in accept_list and '# netout' not in accept_list)) or 
        (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list or '# plmc' not in accept_list or '# pre' not in accept_list or '# dca' not in accept_list or '# err' not in accept_list or '# netout' not in accept_list)) or (len(accept_list) > 2)):
    notxt_flag = False
    # print
    if not os.path.isfile(featurefile):
                print("feature file not exists: ",featurefile, " pass!")   
                return False
  if '# cov' in accept_list:
    if not os.path.isfile(cov):
                print("Cov Matrix file not exists: ",cov, " pass!")
                return False
  if '# plm' in accept_list:
    if not os.path.isfile(plm):
                print("plm matrix file not exists: ",plm, " pass!")
                return False
  if '# plmc' in accept_list:
    if not os.path.isfile(plmc):
                print("plmc matrix file not exists: ",plmc, " pass!")
                return False
  if '# pre' in accept_list:
    if not os.path.isfile(pre):
                print("pre matrix file not exists: ",pre, " pass!")
                return False
  if '# dca' in accept_list:
    if not os.path.isfile(dca):
                print("dca matrix file not exists: ",dca, " pass!")
                return False
  if '# err' in accept_list:
    if not os.path.isfile(err):
                print("err matrix file not exists: ",err, " pass!")
                return False
  if '# netout' in accept_list:      
    if not os.path.isfile(netout):
                print("netout matrix file not exists: ",netout, " pass!")
                return False
  if '# aa' in accept_list:      
    if not os.path.isfile(aa):
                print("aa matrix file not exists: ",aa, " pass!")
                return False

  (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, plmc, pre, dca, err, netout, aa, accept_list, pdb_len, notxt_flag)  
  if featuredata == False or feature_index_all_dict == False:
    print("Bad alignment, Please check!\n")
    return False
  
  ### merge 1D data to L*m
  ### merge 2D data to  L*L*n
  feature_2D_all=[]
  for key in sorted(feature_index_all_dict.keys()):
      featurename = feature_index_all_dict[key]
      feature = featuredata[key]
      feature = np.asarray(feature)
      #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
      
      if feature.shape[0] == feature.shape[1]:
        feature_2D_all.append(feature)
      else:
        print("Wrong dimension")
  
  # fea_len = feature_2D_all[0].shape[0]
  F_2D = len(feature_2D_all)
  # feature_2D_all = np.asarray(feature_2D_all)
  #print(feature_2D_all.shape)
  # print("Total ",F_2D, " 2D features")
  X_2D = np.zeros((xcount, l_max, l_max, F_2D))
  pdb_indx = 0
  for pdb_name in sorted(selected_ids):
      # print(pdb_name, "..",end='')

      featurefile =feature_dir + '/other/' + 'X-'  + sample_pdb + '.txt'
      if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list and '# plmc' not in accept_list and '# pre' not in accept_list and '# dca' not in accept_list and '# err' not in accept_list and '# netout' not in accept_list)) or 
        (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list or '# plmc' not in accept_list or '# pre' not in accept_list or '# dca' not in accept_list or '# err' not in accept_list or '# netout' not in accept_list)) or (len(accept_list) > 2)):
        notxt_flag = False
        if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue   
      cov = feature_dir + '/cov/' + pdb_name + '.cov'  
      if '# cov' in accept_list:
        if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue     
      plm = feature_dir + '/plm/' + pdb_name + '.plm'   
      if '# plm' in accept_list:
        if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue 
      plmc = feature_dir + '/plmc/' + pdb_name + '.plmc'   
      if '# plmc' in accept_list:
        if not os.path.isfile(plmc):
                    print("plmc matrix file not exists: ",plmc, " pass!")
                    continue 
      pre = feature_dir + '/pre/' + pdb_name + '.pre'   
      if '# pre' in accept_list:
        if not os.path.isfile(pre):
                    print("pre matrix file not exists: ",pre, " pass!")
                    continue 
      dca = feature_dir + '/dca/' + pdb_name + '.dca'   
      if '# dca' in accept_list:
        if not os.path.isfile(dca):
                    print("dca matrix file not exists: ",dca, " pass!")
                    continue 
      err = feature_dir + '/dist_error/' + pdb_name + '.txt'   
      if '# err' in accept_list:
        if not os.path.isfile(err):
                    print("err matrix file not exists: ",err, " pass!")
                    continue 
      netout = feature_dir + '/net_out/' + pdb_name + '.npy'
      if '# netout' in accept_list:      
        if not os.path.isfile(netout):
                    print("netout matrix file not exists: ",netout, " pass!")
                    continue 
      aa = feature_dir + '/aa/' + pdb_name + '.aa'
      if '# aa' in accept_list:      
        if not os.path.isfile(aa):
                    print("aa matrix file not exists: ",aa, " pass!")
                    continue 
      ### load the data
      (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, plmc, pre, dca, err, netout, aa, accept_list, pdb_len, notxt_flag)     
      ### merge 1D data to L*m
      ### merge 2D data to  L*L*n
      feature_2D_all=[]
      for key in sorted(feature_index_all_dict.keys()):
          featurename = feature_index_all_dict[key]
          feature = featuredata[key]
          feature = np.asarray(feature)
          #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
          
          if feature.shape[0] == feature.shape[1]:
            feature_2D_all.append(feature)
          else:
            print("Wrong dimension")
     
      L = feature_2D_all[0].shape[0]
      F = len(feature_2D_all)
      X_tmp = np.zeros((L, L, F))
      for i in range (0, F):
        X_tmp[:,:, i] = feature_2D_all[i]      
      
      feature_2D_all = X_tmp
      #print feature_2D_all.shape #(123, 123, 18) 
      if len(feature_2D_all[0, 0, :]) != F_2D:
        print('ERROR! 2D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        exit;

      ### expand to lmax
      if feature_2D_all[0].shape[0] <= l_max:
        # print("extend to lmax: ",feature_2D_all.shape)
        L = feature_2D_all.shape[0]
        F = feature_2D_all.shape[2]
        X_tmp = np.zeros((l_max, l_max, F))
        for i in range (0, F):
          X_tmp[0:L,0:L, i] = feature_2D_all[:,:,i]
        feature_2D_all_complete = X_tmp
      X_2D[pdb_indx, :, :, :] = feature_2D_all_complete
      pdb_indx = pdb_indx + 1
  return X_2D

def get_x_2D_from_this_list_pred(selected_ids, feature_dir, l_max,dist_string, reject_fea_file='None', pdb_len = 0):
  xcount = len(selected_ids)
  sample_pdb = ''
  for pdb in selected_ids:
    sample_pdb = pdb
    break
  accept_list = []
  notxt_flag = True
  if reject_fea_file != 'None':
    with open(reject_fea_file) as f:
      for line in f:
        if line.startswith('#'):
          feature_name = line.strip()
          feature_name = feature_name[0:]
          accept_list.append(feature_name)

  featurefile =feature_dir + '/' + 'X-'  + sample_pdb + '.txt'
  cov =feature_dir + '/' + sample_pdb + '.cov'
  plm =feature_dir + '/' + sample_pdb + '.plm'
  plmc =feature_dir + '/' + sample_pdb + '.plmc'
  pre =feature_dir + '/' + sample_pdb + '.pre'
  dca =feature_dir + '/'  + sample_pdb + '.dca'
  err =feature_dir + '/'  + sample_pdb + '.txt'
  netout = feature_dir + '/' + sample_pdb + '.npy'
  aa = feature_dir + '/' + sample_pdb + '.aa'
  # print(featurefile)
  if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list and '# plmc' not in accept_list and '# pre' not in accept_list and '# dca' not in accept_list and '# err' not in accept_list and '# netout' not in accept_list)) or 
        (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list or '# plmc' not in accept_list or '# pre' not in accept_list or '# dca' not in accept_list or '# err' not in accept_list or '# netout' not in accept_list)) or (len(accept_list) > 2)):
    notxt_flag = False
    # print
    if not os.path.isfile(featurefile):
                print("feature file not exists: ",featurefile, " pass!")   
                return False
  if '# cov' in accept_list:
    if not os.path.isfile(cov):
                print("Cov Matrix file not exists: ",cov, " pass!")
                return False
  if '# plm' in accept_list:
    if not os.path.isfile(plm):
                print("plm matrix file not exists: ",plm, " pass!")
                return False
  if '# plmc' in accept_list:
    if not os.path.isfile(plmc):
                print("plmc matrix file not exists: ",plmc, " pass!")
                return False
  if '# pre' in accept_list:
    if not os.path.isfile(pre):
                print("pre matrix file not exists: ",pre, " pass!")
                return False
  if '# dca' in accept_list:
    if not os.path.isfile(dca):
                print("dca matrix file not exists: ",dca, " pass!")
                return False
  if '# err' in accept_list:
    if not os.path.isfile(err):
                print("err matrix file not exists: ",err, " pass!")
                return False
  if '# netout' in accept_list:      
    if not os.path.isfile(netout):
                print("netout matrix file not exists: ",netout, " pass!")
                return False
  if '# aa' in accept_list:      
    if not os.path.isfile(aa):
                print("aa matrix file not exists: ",aa, " pass!")
                return False

  (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, plmc, pre, dca, err, netout, aa, accept_list, pdb_len, notxt_flag)    

  
  ### merge 1D data to L*m
  ### merge 2D data to  L*L*n
  feature_2D_all=[]
  for key in sorted(feature_index_all_dict.keys()):
      featurename = feature_index_all_dict[key]
      feature = featuredata[key]
      feature = np.asarray(feature)
      #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
      
      if feature.shape[0] == feature.shape[1]:
        feature_2D_all.append(feature)
      else:
        print("Wrong dimension")
  
  # fea_len = feature_2D_all[0].shape[0]
  F_2D = len(feature_2D_all)
  # feature_2D_all = np.asarray(feature_2D_all)
  #print(feature_2D_all.shape)
  # print("Total ",F_2D, " 2D features")
  X_2D = np.zeros((xcount, l_max, l_max, F_2D))
  pdb_indx = 0
  for pdb_name in sorted(selected_ids):
      # print(pdb_name, "..",end='')

      featurefile =feature_dir + '/' + 'X-'  + sample_pdb + '.txt'
      if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list and '# plmc' not in accept_list and '# pre' not in accept_list and '# dca' not in accept_list and '# err' not in accept_list and '# netout' not in accept_list)) or 
        (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list or '# plmc' not in accept_list or '# pre' not in accept_list or '# dca' not in accept_list or '# err' not in accept_list or '# netout' not in accept_list)) or (len(accept_list) > 2)):
        notxt_flag = False
        if not os.path.isfile(featurefile):
                    print("feature file not exists: ",featurefile, " pass!")
                    continue   
      cov = feature_dir + '/' + pdb_name + '.cov'  
      if '# cov' in accept_list:
        if not os.path.isfile(cov):
                    print("Cov Matrix file not exists: ",cov, " pass!")
                    continue     
      plm = feature_dir + '/' + pdb_name + '.plm'   
      if '# plm' in accept_list:
        if not os.path.isfile(plm):
                    print("plm matrix file not exists: ",plm, " pass!")
                    continue 
      plmc = feature_dir + '/' + pdb_name + '.plmc'   
      if '# plmc' in accept_list:
        if not os.path.isfile(plmc):
                    print("plmc matrix file not exists: ",plmc, " pass!")
                    continue 
      pre = feature_dir + '/' + pdb_name + '.pre'   
      if '# pre' in accept_list:
        if not os.path.isfile(pre):
                    print("pre matrix file not exists: ",pre, " pass!")
                    continue 
      dca = feature_dir + '/' + pdb_name + '.dca'   
      if '# dca' in accept_list:
        if not os.path.isfile(dca):
                    print("dca matrix file not exists: ",dca, " pass!")
                    continue 
      err = feature_dir + '/' + pdb_name + '.txt'   
      if '# err' in accept_list:
        if not os.path.isfile(err):
                    print("err matrix file not exists: ",err, " pass!")
                    continue 
      netout = feature_dir + '/' + pdb_name + '.npy'
      if '# netout' in accept_list:      
        if not os.path.isfile(netout):
                    print("netout matrix file not exists: ",netout, " pass!")
                    continue 
      aa = feature_dir + '/' + pdb_name + '.npy'
      if '# aa' in accept_list:      
        if not os.path.isfile(aa):
                    print("aa matrix file not exists: ",aa, " pass!")
                    continue 
      ### load the data
      (featuredata,feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, plmc, pre, dca, err, netout, aa, accept_list, pdb_len, notxt_flag)   
      ### merge 1D data to L*m
      ### merge 2D data to  L*L*n
      feature_2D_all=[]
      for key in sorted(feature_index_all_dict.keys()):
          featurename = feature_index_all_dict[key]
          feature = featuredata[key]
          feature = np.asarray(feature)
          #print("keys: ", key, " featurename: ",featurename, " feature_shape:", feature.shape)
          
          if feature.shape[0] == feature.shape[1]:
            feature_2D_all.append(feature)
          else:
            print("Wrong dimension")
     
      L = feature_2D_all[0].shape[0]
      F = len(feature_2D_all)
      X_tmp = np.zeros((L, L, F))
      for i in range (0, F):
        X_tmp[:,:, i] = feature_2D_all[i]      
      
      feature_2D_all = X_tmp
      #print feature_2D_all.shape #(123, 123, 18) 
      if len(feature_2D_all[0, 0, :]) != F_2D:
        print('ERROR! 2D Feature length of ',sample_pdb,' not equal to ',pdb_name)
        exit;

      ### expand to lmax
      if feature_2D_all[0].shape[0] <= l_max:
        # print("extend to lmax: ",feature_2D_all.shape)
        L = feature_2D_all.shape[0]
        F = feature_2D_all.shape[2]
        X_tmp = np.zeros((l_max, l_max, F))
        for i in range (0, F):
          X_tmp[0:L,0:L, i] = feature_2D_all[:,:,i]
        feature_2D_all_complete = X_tmp
      X_2D[pdb_indx, :, :, :] = feature_2D_all_complete
      pdb_indx = pdb_indx + 1
  return X_2D
  
def evaluate_prediction (dict_l, dict_n, dict_e, P, Y, min_seq_sep):
  P2 = floor_lower_left_to_zero(P, min_seq_sep)
  datacount = len(Y[:, 0])
  L = int(math.sqrt(len(Y[0, :])))
  Y1 = floor_lower_left_to_zero(Y, min_seq_sep)
  list_acc_l5 = []
  list_acc_l2 = []
  list_acc_1l = []
  P3L5 = ceil_top_xL_to_one(dict_l, P2, Y, 0.2)
  P3L2 = ceil_top_xL_to_one(dict_l, P2, Y, 0.5)
  P31L = ceil_top_xL_to_one(dict_l, P2, Y, 1)
  (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l) = print_detailed_evaluations(dict_l, dict_n, dict_e, P3L5, P3L2, P31L, Y)
  return (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)

def evaluate_prediction_4 (dict_l, P, Y, min_seq_sep):
  P2 = floor_lower_left_to_zero(P, min_seq_sep)
  datacount = len(Y[:, 0])
  L = int(math.sqrt(len(Y[0, :])))
  Y1 = floor_lower_left_to_zero(Y, min_seq_sep)
  list_acc_l5 = []
  list_acc_l2 = []
  list_acc_1l = []
  P3L5 = ceil_top_xL_to_one_4(dict_l, P2, Y, 0.2)
  P3L2 = ceil_top_xL_to_one_4(dict_l, P2, Y, 0.5)
  P31L = ceil_top_xL_to_one_4(dict_l, P2, Y, 1)
  avg_prec_l5, avg_prec_l2, avg_prec_1l, avg_mcc_l5, avg_mcc_l2, avg_mcc_1l, avg_recall_l5, avg_recall_l2, avg_recall_1l, avg_f1_l5, avg_f1_l2, avg_f1_1l = print_detailed_evaluations_4(dict_l, P3L5, P3L2, P31L, P2, Y)
  return (avg_prec_l5, avg_prec_l2, avg_prec_1l, avg_mcc_l5, avg_mcc_l2, avg_mcc_1l, avg_recall_l5, avg_recall_l2, avg_recall_1l, avg_f1_l5, avg_f1_l2, avg_f1_1l)

def evaluate_prediction_dist_4(y_pred, y_true):
  
  y_bool = y_true <= 20.0
  y_bool_invert = y_true > 20.0
  y_mean = np.mean(y_true)
  y_pred_below = y_pred * y_bool 
  y_pred_upper = y_pred * y_bool_invert 
  y_true_below = y_true * y_bool 
  y_true_upper = y_true * y_bool_invert 

  weights1 = 1.0
  weights2 = 1/(1 + np.square(y_pred_upper / y_mean))
  global_mse = np.mean(np.square(y_pred - y_true))
  weighted_mse =  np.mean(np.square((y_pred_below - y_true_below))*weights1) + np.mean(np.square((y_pred_upper - y_true_upper))*weights2)
  return global_mse, weighted_mse

def evaluate_prediction_dist_error_4(y_pred, y_true):
  # print("y_pred.shape: ",y_pred.shape)
  # print("y_true.shape: ",y_true.shape)
  y_pred_dist = y_pred[0,:,:,0]
  y_pred_error = y_pred[0,:,:,1]
  y_true_dist = y_true[0,:,:,0]
  y_true_error = y_true[0,:,:,1]
  
  y_bool = y_true_dist <= 20.0
  y_bool_invert = y_true_dist > 20.0
  y_mean = np.mean(y_true_dist)
  y_pred_below = y_pred_dist * y_bool 
  y_pred_upper = y_pred_dist * y_bool_invert 
  y_true_below = y_true_dist * y_bool 
  y_true_upper = y_true_dist * y_bool_invert 

  weights1 = 1.0
  weights2 = 1/(1 + np.square(y_pred_upper / y_mean))
  dist_global_mse = np.mean(np.square(y_pred_dist - y_true_dist))
  dist_weighted_mse =  np.mean(np.square((y_pred_below - y_true_below))*weights1) + np.mean(np.square((y_pred_upper - y_true_upper))*weights2)
  error_global_mse = np.mean(np.square(y_pred_error - y_true_error))
  return dist_global_mse, dist_weighted_mse, error_global_mse

# Floor everything below the triangle of interest to zero
def floor_lower_left_to_zero(XP, min_seq_sep):
  X = np.copy(XP)
  datacount = len(X[:, 0])
  L = int(math.sqrt(len(X[0, :])))
  X_reshaped = X.reshape(datacount, L, L)
  for p in range(0,L):
    for q in range(0,L):
      if ( q - p < min_seq_sep):
        X_reshaped[:, p, q] = 0
  X = X_reshaped.reshape(datacount, L * L)
  return X

# Ceil top xL predictions to 1, others to zero
def ceil_top_xL_to_one(ref_file_dict, XP, Y, x):
  X_ceiled = np.copy(XP)
  i = -1
  for pdb in sorted(ref_file_dict):
    i = i + 1
    xL = int(x * ref_file_dict[pdb])
    X_ceiled[i, :] = np.zeros(len(XP[i, :]))
    X_ceiled[i, np.argpartition(XP[i, :], -xL)[-xL:]] = 1
  return X_ceiled

# Ceil top xL predictions to 1, Length according to the Y
def ceil_top_xL_to_one_4(ref_file_dict, XP, Y, x):
  X_ceiled = np.copy(XP)
  i = -1
  for pdb in sorted(ref_file_dict):
    i = i + 1
    xL = int(x * int(math.sqrt(len(Y[i]))))
    X_ceiled[i, :] = np.zeros(len(XP[i, :]))
    X_ceiled[i, np.argpartition(XP[i, :], -xL)[-xL:]] = 1
  return X_ceiled

def print_detailed_evaluations(dict_l, dict_n, dict_e, PL5, PL2, PL, Y):
  datacount = len(dict_l)
  print("  ID    PDB      L   Nseq   Neff     Nc    L/5  PcL/5  PcL/2   Pc1L    AccL/5    AccL/2      AccL")
  avg_nc  = 0    # average true Nc
  avg_pc_l5  = 0 # average predicted correct L/5
  avg_pc_l2  = 0 # average predicted correct L/2
  avg_pc_1l  = 0 # average predicted correct 1L
  avg_acc_l5 = 0.0
  avg_acc_l2 = 0.0
  avg_acc_1l = 0.0
  list_acc_l5 = []
  list_acc_l2 = []
  list_acc_1l = []
  i = -1
  for pdb in sorted(dict_l):
    i = i + 1
    nc = int(Y[i].sum())
    L = dict_l[pdb]
    L5 = int(L/5)
    L2 = int(L/2)
    pc_l5 = np.logical_and(Y[i], PL5[i, :]).sum()
    pc_l2 = np.logical_and(Y[i], PL2[i, :]).sum()
    pc_1l = np.logical_and(Y[i], PL[i, :]).sum()
    acc_l5 = float(pc_l5) / (float(L5) + epsilon)
    acc_l2 = float(pc_l2) / (float(L2) + epsilon)
    acc_1l = float(pc_1l) / (float(L) + epsilon)
    list_acc_l5.append(acc_l5)
    list_acc_l2.append(acc_l2)
    list_acc_1l.append(acc_1l)
    print(" %3s %6s %6s %6s %6s %6s %6s %6s %6s %6s    %.4f    %.4f    %.4f" % (i, pdb, L, dict_n[pdb], dict_e[pdb], nc, L5, pc_l5, pc_l2, pc_1l, acc_l5, acc_l2, acc_1l))
    avg_nc = avg_nc + nc
    avg_pc_l5 = avg_pc_l5 + pc_l5
    avg_pc_l2 = avg_pc_l2 + pc_l2
    avg_pc_1l = avg_pc_1l + pc_1l
    avg_acc_l5 = avg_acc_l5 + acc_l5
    avg_acc_l2 = avg_acc_l2 + acc_l2
    avg_acc_1l = avg_acc_1l + acc_1l
  avg_nc = int(avg_nc/datacount)
  avg_pc_l5 = int(avg_pc_l5/datacount)
  avg_pc_l2 = int(avg_pc_l2/datacount)
  avg_pc_1l = int(avg_pc_1l/datacount)
  avg_acc_l5 = avg_acc_l5/datacount
  avg_acc_l2 = avg_acc_l2/datacount
  avg_acc_1l = avg_acc_1l/datacount
  print("   Avg                           %6s        %6s %6s %6s    %.4f    %.4f    %.4f" % (avg_nc, avg_pc_l5, avg_pc_l2, avg_pc_1l, avg_acc_l5, avg_acc_l2, avg_acc_1l))
  print ("")
  return (list_acc_l5, list_acc_l2, list_acc_1l,avg_pc_l5,avg_pc_l2,avg_pc_1l,avg_acc_l5,avg_acc_l2,avg_acc_1l)

def print_detailed_evaluations_4(dict_l, PL5, PL2, PL, P, Y):
  datacount = len(dict_l)

  avg_prec_l5 = 0.0
  avg_prec_l2 = 0.0
  avg_prec_1l = 0.0
  avg_recall_l5 = 0.0
  avg_recall_l2 = 0.0
  avg_recall_1l = 0.0
  avg_f1_l5 = 0.0
  avg_f1_l2 = 0.0
  avg_f1_1l = 0.0
  avg_mcc_l5 = 0.0
  avg_mcc_l2 = 0.0
  avg_mcc_1l = 0.0
  i = -1
  for pdb in sorted(dict_l):
    mcc_l5    = matthews_corrcoef(Y[i], PL5[i, :])
    mcc_l2    = matthews_corrcoef(Y[i], PL2[i, :])
    mcc_1l    = matthews_corrcoef(Y[i], PL[i, :])
    prec_l5   = precision_score(Y[i], PL5[i, :])
    prec_l2   = precision_score(Y[i], PL2[i, :])
    prec_1l   = precision_score(Y[i], PL[i, :])
    recall_l5 = recall_score(Y[i], PL5[i, :])
    recall_l2 = recall_score(Y[i], PL2[i, :])
    recall_1l = recall_score(Y[i], PL[i, :])
    F1_l5     = f1_score(Y[i], PL5[i, :])
    F1_l2     = f1_score(Y[i], PL2[i, :])
    F1_1l     = f1_score(Y[i], PL[i, :])
    # pc_l5 = np.logical_and(Y[i], PL5[i, :]).sum()
    # pc_l2 = np.logical_and(Y[i], PL2[i, :]).sum()
    # pc_1l = np.logical_and(Y[i], PL[i, :]).sum()
    # prec_l5 = float(pc_l5) / (float(L5) + epsilon)
    # prec_l2 = float(pc_l2) / (float(L2) + epsilon)
    # prec_1l = float(pc_1l) / (float(L) + epsilon)
    avg_mcc_l5 += mcc_l5
    avg_mcc_l2 += mcc_l2
    avg_mcc_1l += mcc_1l
    avg_prec_l5 += prec_l5
    avg_prec_l2 += prec_l2
    avg_prec_1l += prec_1l
    avg_recall_l5 += recall_l5
    avg_recall_l2 += recall_l2
    avg_recall_1l += recall_1l
    avg_f1_l5 += F1_l5
    avg_f1_l2 += F1_l2
    avg_f1_1l += F1_1l
  avg_mcc_l5 /= datacount
  avg_mcc_l2 /= datacount
  avg_mcc_1l /= datacount
  avg_prec_l5 /= datacount
  avg_prec_l2 /= datacount
  avg_prec_1l /= datacount
  avg_recall_l5 /= datacount
  avg_recall_l2 /= datacount
  avg_recall_1l /= datacount
  avg_f1_l5 /= datacount
  avg_f1_l2 /= datacount
  avg_f1_1l /= datacount
  # print("   Avg                           %6s        %6s %6s %6s    %.4f    %.4f    %.4f" % (avg_nc, avg_pc_l5, avg_pc_l2, avg_pc_1l, avg_prec_l5, avg_prec_l2, avg_prec_1l))
  # print ("")
  return (avg_prec_l5, avg_prec_l2, avg_prec_1l, avg_mcc_l5, avg_mcc_l2, avg_mcc_1l, avg_recall_l5, avg_recall_l2, avg_recall_1l, avg_f1_l5, avg_f1_l2, avg_f1_1l)

