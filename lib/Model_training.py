# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:47:26 2019

@author: Zhiye
"""
import os,sys,gc

from Model_construct import *
from DNCON_lib import *
import numpy as np
import time
import shutil
import pickle
from six.moves import range
import keras.backend as K
import tensorflow as tf
from keras.models import model_from_json,load_model, Sequential, Model
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.utils import multi_gpu_model, Sequence
from keras.callbacks import ReduceLROnPlateau
from random import randint


def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def generate_data_from_file(path_of_lists, path_of_X, path_of_Y, min_seq_sep,dist_string, batch_size, reject_fea_file='None', 
    dist_interval=8, dataset_select='train', feature_2D_num = 441,  if_use_binsize=False, predict_method='bin_class', Maximum_length = 500):
    accept_list = []
    if reject_fea_file != 'None':
        with open(reject_fea_file) as f:
            for line in f:
                if line.startswith('#'):
                    feature_name = line.strip()
                    feature_name = feature_name[0:]
                    accept_list.append(feature_name)
    if (dataset_select == 'train'):
        dataset_list = build_dataset_dictionaries_train(path_of_lists)
    elif (dataset_select == 'vali'):
        dataset_list = build_dataset_dictionaries_test(path_of_lists)
    else:
        dataset_list = build_dataset_dictionaries_train(path_of_lists)

    training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 20000, 'random') #can be random ordered   
    training_list = list(training_dict.keys())
    training_lens = list(training_dict.values())
    all_data_num = len(training_dict)
    loopcount = all_data_num // int(batch_size)
    index = 0
    while(True):
        if index >= loopcount:
            training_dict = subset_pdb_dict(dataset_list, 0, Maximum_length, 20000, 'random') #can be random ordered   
            training_list = list(training_dict.keys())
            training_lens = list(training_dict.values())
            index = 0
        batch_list = training_list[index * batch_size:(index + 1) * batch_size]
        batch_list_len = training_lens[index * batch_size:(index + 1) * batch_size]
        index += 1
        # print(index, end='\t')
        if if_use_binsize:
            max_pdb_lens = Maximum_length
        else:
            max_pdb_lens = max(batch_list_len)

        data_all_dict = dict()
        batch_X  = []
        batch_Y  = []
        batch_Y1 = []
        batch_Y2 = []
        batch_Y3 = []
        batch_Y4 = []
        batch_Y5 = []
        for i in range(0, len(batch_list)):
            pdb_name = batch_list[i]
            pdb_len = batch_list_len[i]
            notxt_flag = True
            featurefile =path_of_X + '/other/' + 'X-'  + pdb_name + '.txt'
            if ((len(accept_list) == 1 and ('# cov' not in accept_list and '# plm' not in accept_list and '# pre' not in accept_list and '# err' not in accept_list and '# netout' not in accept_list)) or 
                  (len(accept_list) == 2 and ('# cov' not in accept_list or '# plm' not in accept_list or '# pre' not in accept_list or '# err' not in accept_list or '# netout' not in accept_list)) or (len(accept_list) > 2)):
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
                    # print("pre matrix file not exists: ",pre, " pass!")
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
                    print("aa matrix file not exists: ",aa, " pass!")
                    continue 

            if predict_method == 'bin_class':       
                targetfile = path_of_Y + '/Y' + str(dist_string) + '-'+ pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif 'mul_class' in predict_method:
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue 
            elif predict_method == 'real_dist':
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  
            elif 'mul_lable' in predict_method:
                targetfile_mul = path_of_Y[0] + pdb_name + '.txt'
                if not os.path.isfile(targetfile_mul):
                        print("target file not exists: ",targetfile_mul, " pass!")
                        continue  
                targetfile_dist = path_of_Y[1] + pdb_name + '.txt'
                if not os.path.isfile(targetfile_dist):
                        print("target file not exists: ",targetfile_dist, " pass!")
                        continue  
            elif 'dist_angle' in predict_method:
                targetfile_mul = path_of_Y[0] + pdb_name + '.txt'
                if not os.path.isfile(targetfile_mul):
                        print("target file not exists: ",targetfile_mul, " pass!")
                        continue  
                targetfile_bin = path_of_Y[1] + pdb_name + '.txt'
                if not os.path.isfile(targetfile_bin):
                        print("target file not exists: ",targetfile_bin, " pass!")
                        continue  
                targetfile_omega = path_of_Y[2] + pdb_name + '.txt'
                if not os.path.isfile(targetfile_omega):
                        print("target file not exists: ",targetfile_omega, " pass!")
                        continue  
                targetfile_theta = path_of_Y[3] + pdb_name + '.txt'
                if not os.path.isfile(targetfile_theta):
                        print("target file not exists: ",targetfile_theta, " pass!")
                        continue  
                targetfile_phi = path_of_Y[4] + pdb_name + '.txt'
                if not os.path.isfile(targetfile_phi):
                        print("target file not exists: ",targetfile_phi, " pass!")
                        continue  
            else:
                targetfile = path_of_Y + pdb_name + '.txt'
                if not os.path.isfile(targetfile):
                        print("target file not exists: ",targetfile, " pass!")
                        continue  

            (featuredata, feature_index_all_dict) = getX_2D_format(featurefile, cov, plm, plmc, pre, dca, err, netout, aa, accept_list, pdb_len, notxt_flag)
            if featuredata == False or feature_index_all_dict == False:
                print("Bad alignment, Please check!\n")
                continue
            feature_2D_all = []
            for key in sorted(feature_index_all_dict.keys()):
                featurename = feature_index_all_dict[key]
                feature = featuredata[key]
                feature = np.asarray(feature)
                if feature.shape[0] == feature.shape[1]:
                    feature_2D_all.append(feature)
                else:
                    print("Wrong dimension")
            fea_len = feature_2D_all[0].shape[0]

            F = len(feature_2D_all)
            if F != feature_2D_num:
                print("Target %s has wrong feature shape! Continue!" % pdb_name)
                continue
            X = np.zeros((max_pdb_lens, max_pdb_lens, F))
            for m in range(0, F):
                X[0:fea_len, 0:fea_len, m] = feature_2D_all[m]
            # X = np.memmap(cov, dtype=np.float32, mode='r', shape=(F, max_pdb_lens, max_pdb_lens))
            # X = X.transpose(1, 2, 0)
            l_max = max_pdb_lens
            if predict_method == 'bin_class':
                Y = getY(targetfile, min_seq_sep, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            elif predict_method == 'mul_class_R':
                Y1 = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=25 #21 42
                Y= (np.arange(max_class) == Y1[...,None]).astype(int) 
                Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) 
            elif predict_method == 'mul_class_D':
                Y1 = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=33 #21 42
                Y= (np.arange(max_class) == Y1[...,None]).astype(int) 
                Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) 
            elif predict_method == 'mul_class_T':
                Y1 = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=38 #21 42
                Y= (np.arange(max_class) == Y1[...,None]).astype(int) 
                Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) 
            elif predict_method == 'mul_class_G':
                Y1 = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=42 
                Y= (np.arange(max_class) == Y1[...,None]).astype(int) 
                Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3])
            elif predict_method == 'mul_class_C':
                Y1 = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=10 
                Y= (np.arange(max_class) == Y1[...,None]).astype(int) 
                Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3])
            elif predict_method == 'real_dist':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
            elif predict_method == 'real_dist_limited16':
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>16] = 16
                # normalization [0,16]
                Y = Y/16.0
            elif 'real_dist_limited' in predict_method and predict_method[len('real_dist_limited'):] != '': #real_dist_limited16
                dist_threshold = int(predict_method[len('real_dist_limited'):])
                Y = getY(targetfile, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1)
                Y[Y>dist_threshold] = dist_threshold
            elif predict_method == 'mul_lable':
                Y1 = getY(targetfile_mul, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=18  #3.5-19 21 3.5-16 18 3.5-24 26
                Y= (np.arange(max_class) == Y1[...,None]).astype(int) 
                
                Y_mul = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) 

                Y = getY(targetfile_dist, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y_dist = Y.reshape(l_max, l_max, 1)
                Y[Y>16] = 16  #3.5-19 21 3.5-16 18 3.5-24 26
            elif predict_method == 'mul_lable_R':
                Y1 = getY(targetfile_mul, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=25  #3.5-19 21 3.5-16 18 3.5-24 26 4.5-16 25
                Y= (np.arange(max_class) == Y1[...,None]).astype(int) 
                
                Y_mul = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) 

                Y = getY(targetfile_dist, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y_dist = Y.reshape(l_max, l_max, 1)
                Y_dist[Y_dist>16] = 16  #3.5-19 21 3.5-16 18 3.5-24 26 4.5-16 25
            elif predict_method == 'mul_lable_D':
                Y1 = getY(targetfile_mul, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=33  #3.5-19 21 3.5-16 18 3.5-24 26 4.5-16 25
                Y= (np.arange(max_class) == Y1[...,None]).astype(int)
                
                Y_mul = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) 

                Y = getY(targetfile_dist, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y_dist = Y.reshape(l_max, l_max, 1)
                Y[Y>19] = 19  #3.5-19 21 3.5-16 18 3.5-24 26 4.5-16 25
            elif predict_method == 'mul_lable_G':
                Y1 = getY(targetfile_mul, 0, l_max)
                if (l_max * l_max != len(Y1)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y1 = Y1.reshape(l_max, l_max, 1) #contains class id
                max_class=42  #3.5-19 21 3.5-16 18 3.5-24 26 4.5-16 25 2-22 42
                Y= (np.arange(max_class) == Y1[...,None]).astype(int)
                
                Y_mul = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) 

                Y = getY(targetfile_dist, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y_dist = Y.reshape(l_max, l_max, 1)
                Y[Y>22] = 22  #3.5-19 21 3.5-16 18 3.5-24 26 4.5-16 25
            elif predict_method == 'dist_angle1':
                Y = getY(targetfile_bin, 0, l_max) #targetfile_bin is real dist file
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y[Y == 0] = 100
                Y[Y <= int(dist_interval)] = 1
                Y[Y > int(dist_interval)] = 0
                Y_bin = Y.reshape(l_max, l_max, 1)
                Y_no_contact = 1-Y_bin

                Y = getY(targetfile_mul, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1) #contains class id
                max_class=38  
                Y = (np.arange(max_class) == Y[...,None]).astype(int)
                Y_mul = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[3]) 
                Y_mul = np.concatenate((Y_no_contact, Y_mul[:,:,1:37]), axis=-1) # after this distance channel is 37

                Y = getY(targetfile_omega, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1) #contains class id
                Y = (np.arange(24) == Y[...,None]).astype(int)
                Y_omega = Y.reshape(Y.shape[0], Y.shape[1], Y.shape[3])*Y_bin
                Y_omega = np.concatenate((Y_no_contact, Y_omega), axis=-1)

                Y = getY(targetfile_theta, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1) #contains class id
                Y = (np.arange(24) == Y[...,None]).astype(int)
                Y_theta = Y.reshape(Y.shape[0], Y.shape[1], Y.shape[3])*Y_bin
                Y_theta = np.concatenate((Y_no_contact, Y_theta), axis=-1)

                Y = getY(targetfile_phi, 0, l_max)
                if (l_max * l_max != len(Y)):
                    print('Error!! y does not have L * L feature values!!, pdb_name = %s'%(pdb_name))
                    continue
                Y = Y.reshape(l_max, l_max, 1) #contains class id
                Y = (np.arange(12) == Y[...,None]).astype(int)
                Y_phi = Y.reshape(Y.shape[0], Y.shape[1], Y.shape[3])*Y_bin
                Y_phi = np.concatenate((Y_no_contact, Y_phi), axis=-1)

            if 'mul_lable' in predict_method:
                batch_X.append(X)
                batch_Y1.append(Y_mul)
                batch_Y2.append(Y_dist)
                del X, Y_mul, Y_dist
            elif 'dist_angle' in predict_method:
                batch_X.append(X)
                batch_Y1.append(Y_mul)
                batch_Y2.append(Y_omega)
                batch_Y3.append(Y_theta)
                batch_Y4.append(Y_phi)
                del X, Y_mul, Y_omega, Y_theta, Y_phi
            else:
                batch_X.append(X)
                batch_Y.append(Y)
                del X, Y
        # print(predict_method)
        if 'mul_lable' in predict_method:
            batch_X =  np.array(batch_X)
            batch_Y1 =  np.array(batch_Y1)
            batch_Y2 =  np.array(batch_Y2)
            if len(batch_X.shape) < 4 or len(batch_Y1.shape) < 4 or len(batch_Y2.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield batch_X, [batch_Y1, batch_Y2]
        elif 'dist_angle' in predict_method:
            batch_X =  np.array(batch_X)
            batch_Y1 =  np.array(batch_Y1)
            batch_Y2 =  np.array(batch_Y2)
            batch_Y3 =  np.array(batch_Y3)
            batch_Y4 =  np.array(batch_Y4)
            if len(batch_X.shape) < 4 or len(batch_Y1.shape) < 4 or len(batch_Y2.shape) < 4 or len(batch_Y3.shape) < 4 or len(batch_Y4.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield batch_X, [batch_Y1, batch_Y2, batch_Y3, batch_Y4]
        else:
            batch_X =  np.array(batch_X)
            batch_Y =  np.array(batch_Y)
            # print('X shape\n', batch_X.shape)
            # print('Y shape', batch_Y.shape)
            if len(batch_X.shape) < 4 or len(batch_Y.shape) < 4:
                print('Data shape error, pass!\n')
                continue
            yield batch_X, batch_Y

def DeepDist_train_generator(feature_num,CV_dir, model_prefix,
    epoch_outside,epoch_inside,epoch_rerun,win_array,nb_filters,nb_layers, batch_size_train,path_of_lists, path_of_Y, path_of_X, Maximum_length,dist_string, reject_fea_file='None',
    initializer = "he_normal", loss_function = "binary_crossentropy", runcount=1.0,  dist_interval=8,  if_use_binsize = False, if_use_generator=True, weight = 1.0): 

    print("\n######################################\n佛祖保佑，永不迨机，永无bug，精度九十九\n######################################\n")
    feature_2D_num=feature_num # the number of features for each residue
 
    print("Load feature number", feature_2D_num)
    ### Define the model 
    model_out= "%s/model-train-%s.json" % (CV_dir,model_prefix)
    model_weight_out = "%s/model-train-weight-%s.h5" % (CV_dir,model_prefix)
    model_weight_out_best = "%s/model-train-weight-%s-best-val.h5" % (CV_dir,model_prefix)

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000)#0.001  decay=0.0
    if model_prefix == 'DEEPDIST_RESRC':
        DeepDist = DeepDistRes_with_paras_2D(win_array,feature_2D_num, nb_filters,nb_layers,opt,initializer,loss_function)
    else:
        DeepDist = DeepDistRes_with_paras_2D(win_array,feature_2D_num, nb_filters,nb_layers,opt,initializer,loss_function)

    model_json = DeepDist.to_json()
    print("Saved model to disk")
    with open(model_out, "w") as json_file:
        json_file.write(model_json)

    rerun_flag=0
    train_acc_history_out = "%s/training.acc_history" % (CV_dir)
    val_acc_history_out = "%s/validation.acc_history" % (CV_dir)
    best_val_acc_out = "%s/best_validation.acc_history" % (CV_dir)
    if os.path.exists(model_weight_out_best):
        print("######## Loading existing weights ",model_weight_out_best)
        DeepDist.load_weights(model_weight_out_best)
        rerun_flag = 1
    else:
        print("######## Setting initial weights")   
        chkdirs(train_acc_history_out)     
        with open(train_acc_history_out, "a") as myfile:
          myfile.write("Epoch_outside\tEpoch_inside\tavg_TP_counts_l5\tavg_TP_counts_l2\tavg_TP_counts_1l\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\tGloable_MSE\tWeighted_MSE\n")
          
        chkdirs(val_acc_history_out)     
        with open(val_acc_history_out, "a") as myfile:
          myfile.write("Epoch\tprec_l5\tprec_l2\tprec_1l\tmcc_l5\tmcc_l2\tmcc_1l\trecall_l5\trecall_l2\trecall_1l\tf1_l5\tf1_l2\tf1_1l\n")
        
        chkdirs(best_val_acc_out)     
        with open(best_val_acc_out, "a") as myfile:
          myfile.write("Seq_Name\tSeq_Length\tAvg_Accuracy_l5\tAvg_Accuracy_l2\tAvg_Accuracy_1l\n")

    #predict_method has three value : bin_class, mul_class, real_dist
    if loss_function == 'binary_crossentropy':
        predict_method = 'bin_class'
        path_of_Y_train = path_of_Y + '/bin_class/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'binary_crossentropy'
    elif loss_function == 'weighted_MSE_limited16':
        predict_method = 'real_dist_limited16'
        path_of_Y_train = path_of_Y + '/real_dist/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = _weighted_mean_squared_error(1)
    elif loss_function == 'categorical_crossentropy_R':
        predict_method = 'mul_class_R'
        path_of_Y_train = path_of_Y + '/mul_class_4.5_16/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'categorical_crossentropy'
    elif loss_function == 'categorical_crossentropy_D':
        predict_method = 'mul_class_D'
        path_of_Y_train = path_of_Y + '/mul_class_3.5_19/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'categorical_crossentropy'
    elif loss_function == 'categorical_crossentropy_G':
        predict_method = 'mul_class_G'
        path_of_Y_train = path_of_Y + '/mul_class_2_22/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'categorical_crossentropy'
    elif loss_function == 'categorical_crossentropy_T':
        predict_method = 'mul_class_T'
        path_of_Y_train = path_of_Y + '/mul_class_2_20/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'categorical_crossentropy'
    elif loss_function == 'categorical_crossentropy_C':
        predict_method = 'mul_class_C'
        path_of_Y_train = path_of_Y + '/mul_class_4_20/'
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = 'categorical_crossentropy'
    elif loss_function == 'mul_class_and_real_dist_R': # raptorX 4.5-16
        predict_method = 'mul_lable_R'
        path_of_Y_train = [path_of_Y + '/mul_class_4.5_16/', path_of_Y + '/real_dist/']
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = {'mul_class':'categorical_crossentropy', 'real_dist':'mean_squared_error'}
        loss_weight = {'mul_class':weight, 'real_dist':1.0}
    elif loss_function == 'mul_class_and_real_dist_D': # DMPfold 3.5-19
        predict_method = 'mul_lable_D'
        path_of_Y_train = [path_of_Y + '/mul_class_3.5_19/', path_of_Y + '/real_dist/']
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = {'mul_class':'categorical_crossentropy', 'real_dist':'mean_squared_error'}
        loss_weight = {'mul_class':weight, 'real_dist':1.0}
    elif loss_function == 'mul_class_and_real_dist_G': # Google 2-22
        predict_method = 'mul_lable_G'
        path_of_Y_train = [path_of_Y + '/mul_class_2_22/', path_of_Y + '/real_dist/']
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = {'mul_class':'categorical_crossentropy', 'real_dist':'mean_squared_error'}
        loss_weight = {'mul_class':weight, 'real_dist':1.0}
    elif loss_function == 'dist_angle1': # 
        predict_method = 'dist_angle1'
        path_of_Y_train = [path_of_Y + '/mul_class_2_20/', path_of_Y + '/real_dist/', path_of_Y + '/omega/', path_of_Y + '/theta/', path_of_Y + '/phi/']
        path_of_Y_evalu = path_of_Y + '/bin_class/'
        loss_function = {'dist':'categorical_crossentropy', 'omega':'categorical_crossentropy', 'theta':'categorical_crossentropy', 'phi':'categorical_crossentropy'}
        loss_weight = {'dist':1.0, 'omega':1.0, 'theta':1.0, 'phi':1.0}
    
    if 'mul_lable' in predict_method or 'dist_angle' in predict_method:
        DeepDist.compile(loss=loss_function, loss_weights = loss_weight, metrics=['acc'], optimizer=opt)
    else:
        DeepDist.compile(loss=loss_function, metrics=['acc'], optimizer=opt)


    model_weight_epochs = "%s/model_weights/"%(CV_dir)
    model_weights_top = "%s/model_weights_top/"%(CV_dir)
    model_predict= "%s/predict_map/"%(CV_dir)
    model_val_acc= "%s/val_acc_inepoch/"%(CV_dir)
    chkdirs(model_weight_epochs)
    chkdirs(model_weights_top)
    chkdirs(model_predict)
    chkdirs(model_val_acc)

    tr_l = build_dataset_dictionaries_train(path_of_lists)
    tr_l_dict = subset_pdb_dict(tr_l, 0, Maximum_length, 200000, 'ordered')
    te_l = build_dataset_dictionaries_test(path_of_lists)
    te_l_dict = subset_pdb_dict(te_l, 0, Maximum_length, 200000, 'ordered')
    all_l = te_l.copy()
    train_data_num = len(tr_l_dict)
    child_list_num = int(train_data_num/15)# 15 is the inter
    print('Total Number of Training dataset = ',str(len(tr_l_dict)))
    print('Total Number of Validation dataset = ',str(len(te_l_dict)))

    # callbacks=[reduce_lr]
    train_avg_acc_l5_best = 0 
    val_avg_acc_l5_best = 0
    val_avg_acc_l2_best = 0
    min_seq_sep = 0
    lr_decay = False
    train_loss_last = 1e32
    train_loss_list = []
    mul_loss_list = []
    omega_loss_list = []
    theta_loss_list = []
    phi_loss_list = []
    dist_loss_list = []
    dist2_loss_list = []
    evalu_loss_list = []

    for epoch in range(epoch_rerun,epoch_outside):
        if (epoch >=30 and lr_decay == False):
            print("Setting lr_decay as true")
            lr_decay = True
            if predict_method != 'mul_lable_G' and train_data_num < 10000:#and feature_2D_num > 500
                opt = SGD(lr=0.01, momentum=0.9, decay=0.00, nesterov=True)#0.001
                DeepDist.compile(loss=loss_function, metrics=['accuracy'], optimizer=opt)

        os.system('cp %s %s'%(reject_fea_file, CV_dir))
        print("\n############ Running epoch ", epoch)
        if epoch == 0 and rerun_flag == 0:
            first_inepoch = 1
            history = DeepDist.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, dist_interval = dist_interval, feature_2D_num = feature_2D_num, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
            steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = first_inepoch, max_queue_size=20, workers=2, use_multiprocessing=False)         
            if predict_method == 'mul_lable':
                mul_loss_list.append(history.history['mul_class_loss'][first_inepoch-1])  
                dist_loss_list.append(history.history['real_dist_loss'][first_inepoch-1])  
            elif 'dist_angle' in predict_method:
                mul_loss_list.append(history.history['dist_loss'][first_inepoch-1])  
                omega_loss_list.append(history.history['omega_loss'][first_inepoch-1])  
                theta_loss_list.append(history.history['theta_loss'][first_inepoch-1])  
                phi_loss_list.append(history.history['phi_loss'][first_inepoch-1])  
            else:
                train_loss_list.append(history.history['loss'][first_inepoch-1])
        else: 
            history = DeepDist.fit_generator(generate_data_from_file(path_of_lists, path_of_X, path_of_Y_train, min_seq_sep, dist_string, batch_size_train, reject_fea_file, dist_interval = dist_interval, feature_2D_num = feature_2D_num, if_use_binsize=if_use_binsize, predict_method=predict_method, Maximum_length = Maximum_length), 
                steps_per_epoch = len(tr_l_dict)//batch_size_train, epochs = 1, max_queue_size=20, workers=2, use_multiprocessing=False)        
            if predict_method == 'mul_lable':
                mul_loss_list.append(history.history['mul_class_loss'][0])  
                dist_loss_list.append(history.history['real_dist_loss'][0])   
            elif 'dist_angle' in predict_method:
                mul_loss_list.append(history.history['dist_loss'][0])  
                omega_loss_list.append(history.history['omega_loss'][0])  
                theta_loss_list.append(history.history['theta_loss'][0])  
                phi_loss_list.append(history.history['phi_loss'][0])  
            else:    
                train_loss_list.append(history.history['loss'][0])
 
        DeepDist.save_weights(model_weight_out)
        
        ### save models
        model_weight_out_inepoch = "%s/model-train-weight-%s-epoch%i.h5" % (model_weight_epochs,model_prefix,epoch)
        DeepDist.save_weights(model_weight_out_inepoch)
        ##### running validation

        print("Now evaluate for epoch ",epoch)
        val_acc_out_inepoch = "%s/validation_epoch%i.acc_history" % (model_val_acc, epoch) 
        sys.stdout.flush()
        selected_list = subset_pdb_dict(te_l,   0, Maximum_length, 20000, 'ordered')  ## here can be optimized to automatically get maxL from selected dataset
        print('Loading data sets ..',end='')

        testdata_len_range=50
        step_num = 0
        out_avg_prec_l5 = 0.0
        out_avg_prec_l2 = 0.0
        out_avg_prec_1l = 0.0
        out_avg_mcc_l5 = 0.0
        out_avg_mcc_l2 = 0.0
        out_avg_mcc_1l = 0.0
        out_avg_recall_l5 = 0.0
        out_avg_recall_l2 = 0.0
        out_avg_recall_1l = 0.0
        out_avg_f1_l5 = 0.0
        out_avg_f1_l2 = 0.0
        out_avg_f1_1l = 0.0
        out_gloable_error_mse = 0.0
        out_gloable_mse = 0.0
        out_weighted_mse = 0.0

        print(("SeqName\tSeqLen\tFea\tprec_l5\tprec_l2\tprec_1l\tmcc_l5\tmcc_l2\tmcc_1l\tGloable_MSE\tWeighted_MSE\tGlobal_Error_MSE\n"))

        for key in selected_list:
            value = selected_list[key]
            p1 = {key: value}
            if if_use_binsize:
                length = Maximum_length
            else:
                length = value
            # print(len(p1))
            if len(p1) < 1:
                continue

            selected_list_2D = get_x_2D_from_this_list(p1, path_of_X, length,dist_string, reject_fea_file, value)
            if type(selected_list_2D) == bool:
                continue
            selected_list_label = get_y_from_this_list(p1, path_of_Y_evalu, 0, length, dist_string)# dist_string 80
            if type(selected_list_label) == bool:
                continue
            DeepDist.load_weights(model_weight_out)

            DeepDist_prediction = DeepDist.predict([selected_list_2D], batch_size= 1)

            global_mse = 0.0
            weighted_mse = 0.0
            error_global_mse = 0.0
            if predict_method == 'mul_class' or predict_method == 'mul_class_R':
                DeepDist_prediction= DeepDist_prediction[:,:,:,0:8].sum(axis=-1)
            elif predict_method == 'mul_class_D':
                DeepDist_prediction= DeepDist_prediction[:,:,:,0:10].sum(axis=-1)
            elif predict_method == 'mul_class_T':
                DeepDist_prediction= DeepDist_prediction[:,:,:,0:13].sum(axis=-1)
            elif predict_method == 'mul_class_G':
                DeepDist_prediction= DeepDist_prediction[:,:,:,0:13].sum(axis=-1)
            elif predict_method == 'mul_class_C':
                DeepDist_prediction= DeepDist_prediction[:,:,:,0:3].sum(axis=-1)

            elif predict_method == 'real_dist':
                DeepDist_prediction[DeepDist_prediction>100] = 100 # incase infinity
                DeepDist_prediction[DeepDist_prediction<=0] = 1 # incase infinity
                DeepDist_prediction_dist = np.copy(DeepDist_prediction)
                DeepDist_prediction = 1/DeepDist_prediction # convert to confidence
            elif predict_method == 'real_dist_limited16':
                # selected_list_label_dist = get_y_from_this_list(p1, path_of_Y_train, 0, length, dist_string, lable_type = 'real')# dist_string 80
                # global_mse, weighted_mse = evaluate_prediction_dist_4(DeepDist_prediction, selected_list_label_dist)
                DeepDist_prediction = DeepDist_prediction*16.0
                DeepDist_prediction[DeepDist_prediction>100] = 100 # incase infinity
                DeepDist_prediction[DeepDist_prediction<=0] = 1 # incase infinity
                DeepDist_prediction_dist = np.copy(DeepDist_prediction)
                DeepDist_prediction = 1/DeepDist_prediction # convert to confidence
            elif predict_method == 'mul_lable' or predict_method == 'mul_lable_R':
                DeepDist_prediction_mul_class= DeepDist_prediction[0][:,:,:,0:8].sum(axis=-1)
                DeepDist_prediction_real_dist= DeepDist_prediction[1]
                DeepDist_prediction_real_dist[DeepDist_prediction_real_dist>100] = 100 
                DeepDist_prediction_real_dist[DeepDist_prediction_real_dist<=0] = 1
                DeepDist_prediction_dist = np.copy(DeepDist_prediction_real_dist)

                DeepDist_prediction_dist = 1/DeepDist_prediction_dist
                DeepDist_prediction = (DeepDist_prediction_mul_class.reshape(length, length) + DeepDist_prediction_dist.reshape(length, length))/2.0
            elif predict_method == 'mul_lable_D':
                DeepDist_prediction_mul_class= DeepDist_prediction[0][:,:,:,0:10].sum(axis=-1)
                DeepDist_prediction_real_dist= DeepDist_prediction[1]
                DeepDist_prediction_real_dist[DeepDist_prediction_real_dist>100] = 100 
                DeepDist_prediction_real_dist[DeepDist_prediction_real_dist<=0] = 1
                DeepDist_prediction_dist = np.copy(DeepDist_prediction_real_dist)

                DeepDist_prediction_dist = 1/DeepDist_prediction_dist
                DeepDist_prediction = (DeepDist_prediction_mul_class.reshape(length, length) + DeepDist_prediction_dist.reshape(length, length))/2.0
            elif predict_method == 'mul_lable_G':
                DeepDist_prediction_mul_class= DeepDist_prediction[0][:,:,:,0:13].sum(axis=-1)
                DeepDist_prediction_real_dist= DeepDist_prediction[1]
                DeepDist_prediction_real_dist[DeepDist_prediction_real_dist>100] = 100 
                DeepDist_prediction_real_dist[DeepDist_prediction_real_dist<=0] = 1
                DeepDist_prediction_dist = np.copy(DeepDist_prediction_real_dist)

                DeepDist_prediction_dist = 1/DeepDist_prediction_dist
                DeepDist_prediction = (DeepDist_prediction_mul_class.reshape(length, length) + DeepDist_prediction_dist.reshape(length, length))/2.0
            elif predict_method == 'dist_angle1': # first try only evalu dist
                DeepDist_prediction_mul_class= DeepDist_prediction[0][:,:,:,1:13].sum(axis=-1)
                DeepDist_prediction_omega= DeepDist_prediction[1]
                DeepDist_prediction_theta= DeepDist_prediction[2]
                DeepDist_prediction_phi= DeepDist_prediction[3]

                # DeepDist_prediction_dist = 1/DeepDist_prediction_dist
                DeepDist_prediction = DeepDist_prediction_mul_class.reshape(length, length)

            CMAP = DeepDist_prediction.reshape(length, length)
            Map_UpTrans = (np.triu(CMAP, 1).T + np.tril(CMAP, -1))/2
            Map_UandL = (np.triu(CMAP) + np.tril(CMAP).T)/2
            real_cmap = Map_UandL + Map_UpTrans

            DeepDist_prediction = real_cmap.reshape(len(p1), length*length)
            (avg_prec_l5, avg_prec_l2, avg_prec_1l, avg_mcc_l5, avg_mcc_l2, avg_mcc_1l, avg_recall_l5, avg_recall_l2, avg_recall_1l, avg_f1_l5, avg_f1_l2, avg_f1_1l) = evaluate_prediction_4(p1, DeepDist_prediction, selected_list_label, 24)
            val_acc_history_content = "%s\t%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (key,value, avg_prec_l5,avg_prec_l2,avg_prec_1l, 
                avg_mcc_l5, avg_mcc_l2, avg_mcc_1l, global_mse, weighted_mse, error_global_mse)
            print(val_acc_history_content)
            with open(val_acc_out_inepoch, "a") as myfile:
                myfile.write(val_acc_history_content)

            out_gloable_mse += global_mse
            out_weighted_mse += weighted_mse 
            out_gloable_error_mse += error_global_mse            
            out_avg_prec_l5 += avg_prec_l5 * len(p1)
            out_avg_prec_l2 += avg_prec_l2 * len(p1)
            out_avg_prec_1l += avg_prec_1l * len(p1)
            out_avg_mcc_l5 += avg_mcc_l5 * len(p1)
            out_avg_mcc_l2 += avg_mcc_l2 * len(p1)
            out_avg_mcc_1l += avg_mcc_1l * len(p1)
            out_avg_recall_l5 += avg_recall_l5 * len(p1)
            out_avg_recall_l2 += avg_recall_l2 * len(p1)
            out_avg_recall_1l += avg_recall_1l * len(p1)
            out_avg_f1_l5 += avg_f1_l5 * len(p1)
            out_avg_f1_l2 += avg_f1_l2 * len(p1)
            out_avg_f1_1l += avg_f1_1l * len(p1)
            
            step_num += 1
        print ('step_num=', step_num)
        all_num = len(selected_list)
        out_avg_prec_l5 /= all_num
        out_avg_prec_l2 /= all_num
        out_avg_prec_1l /= all_num
        out_avg_mcc_l5 /= all_num
        out_avg_mcc_l2 /= all_num
        out_avg_mcc_1l /= all_num
        out_avg_recall_l5 /= all_num
        out_avg_recall_l2 /= all_num
        out_avg_recall_1l /= all_num
        out_avg_f1_l5 /= all_num
        out_avg_f1_l2 /= all_num
        out_avg_f1_1l /= all_num
        val_acc_history_content = "%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (epoch, out_avg_prec_l5,out_avg_prec_l2,out_avg_prec_1l, 
            out_avg_mcc_l5, out_avg_mcc_l2, out_avg_mcc_1l, out_gloable_mse, out_weighted_mse, out_gloable_error_mse)

        with open(val_acc_history_out, "a") as myfile:
            myfile.write(val_acc_history_content)  
            myfile.write('\n')

        print('The validation accuracy is ',val_acc_history_content)
        if out_avg_prec_l2 >= val_avg_acc_l2_best:
            val_avg_acc_l2_best = out_avg_prec_l2 
            score_imed = "Accuracy L2 of Val: %.4f\t\n" % (val_avg_acc_l2_best)
            print("Saved best weight to disk, ", score_imed)
            DeepDist.save_weights(model_weight_out_best)

        if (lr_decay and epoch > 30):
            current_lr = K.get_value(DeepDist.optimizer.lr)
            print("Current learning rate is {} ...".format(current_lr))
            if (epoch % 20 == 0):
                K.set_value(DeepDist.optimizer.lr, current_lr * 0.1)
                print("Decreasing learning rate to {} ...".format(current_lr * 0.1))
        if predict_method == 'mul_lable':
            print("Mul class loss history:", mul_loss_list)
            print("Real dist loss history:", dist_loss_list)
        elif predict_method == 'dist_angle':
            print("Dist loss history:", mul_loss_list)
            print("Omega loss history:", omega_loss_list)
            print("Omega loss history:", theta_loss_list)
            print("Omega loss history:", phi_loss_list)
        else:        
            print("Train loss history:", train_loss_list)
        # print("Validation loss history:", evalu_loss_list)
        #clear memory
        # K.clear_session()
        # tf.reset_default_graph()
    #select top10 models
    epochs = []
    accL5s = []
    with open(val_acc_history_out) as f:
        for line in f:
            cols = line.strip().split()
            if cols[0] != '150':
                continue
            else:
                epoch = cols[1]
                accL5 = cols[6]
                epochs.append(cols[1])
                accL5s.append(cols[6])
                # print(epoch, accL5)
    accL5_sort = accL5s.copy()
    accL5_sort.sort(reverse=True)
    accL5_top = accL5_sort[0:5]
    epoch_top = []
    for index in range(len(accL5_top)):
        acc_find = accL5_top[index]
        pos_find = [i for i, v in enumerate(accL5s) if v == acc_find]
        # print(pos_find)
        for num in range(len(pos_find)):
            epoch_top.append(epochs[pos_find[num]])
    epoch_top = list(set(epoch_top))
    for index in range(len(epoch_top)):
        model_weight = "model-train-weight-%s-epoch%i.h5" % (model_prefix,int(epoch_top[index]))
        src_file = os.path.join(model_weight_epochs,model_weight)
        dst_file = os.path.join(model_weights_top,model_weight)
        shutil.copyfile(src_file,dst_file)
        print("Copy %s to model_weights_top"%epoch_top[index])
    print("Training finished, best validation acc = ",val_avg_acc_l2_best)
    return val_avg_acc_l2_best
