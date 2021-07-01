import sys
import os
from shutil import copyfile
import platform
from glob import glob

if len(sys.argv) != 15:
  print('please input the right parameters')
  sys.exit(1)
current_os_name = platform.platform()
print('%s' % current_os_name)

if 'Ubuntu' in current_os_name.split('-'): #on local
  sysflag='local'
elif 'centos' in current_os_name.split('-'): #on lewis or multicom
  sysflag='lewis'

GLOBAL_PATH=os.path.dirname(os.path.dirname(__file__)) #this will auto get the DNCON4 folder name

sys.path.insert(0, GLOBAL_PATH+'/lib/')
print (GLOBAL_PATH)
from Model_training import *
from training_strategy import *
from DNCON_lib import *


net_name = str(sys.argv[1]) # DNCON4_RES
dataset = str(sys.argv[2])  # DNCON2, DNCON4, DEEPCOV, RESPRE
fea_file = str(sys.argv[3])
loss_function = str(sys.argv[4]) # binary_crossentropy weighted_MSE weighted_MSE_limited16 sigmoid_MSE categorical_crossentropy weighted_MSElimited20_disterror
nb_filters=int(sys.argv[5]) 
nb_layers=int(sys.argv[6]) 
filtsize=int(sys.argv[7]) 
out_epoch=int(sys.argv[8])
in_epoch=int(sys.argv[9]) 
feature_dir = sys.argv[10] 
outputdir = sys.argv[11] 
acclog_dir = sys.argv[12]
weight = float(sys.argv[13])
index = float(sys.argv[14])


CV_dir=outputdir+'/'+net_name+'_'+dataset+'_'+fea_file + '_' + loss_function + '_filter'+str(nb_filters)+'_layers'+str(nb_layers)+'_ftsize'+str(filtsize)+'_'+str(index)

lib_dir=GLOBAL_PATH+'/lib/'

gpu_mem = gpu_schedul_strategy(sysflag, gpu_mem_rate = 0.9, allow_growth = False)

rerun_epoch=0
if not os.path.exists(CV_dir):
  os.makedirs(CV_dir)
else:
  h5_num = len(glob(CV_dir + '/model_weights/*.h5'))
  rerun_epoch = h5_num
  if rerun_epoch <= 0:
    rerun_epoch = 0
    print("This parameters already exists, quit")
    # sys.exit(1)
  print("####### Restart at epoch ", rerun_epoch)

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def chkfiles(fn):
  if os.path.exists(fn):
    return True 
  else:
    return False

dist_string = '80'
if dataset == 'DEEPDIST':
  path_of_lists   = GLOBAL_PATH+'/data/DEEPDIST/lists-test-train/'
else:
  #add your dataset path
  print('Please input the dataset parameter!')
  sys.exit(1)
reject_fea_file = GLOBAL_PATH+'/lib/feature_txt/'+fea_file+'.txt'
path_of_Y       = feature_dir 
path_of_X       = feature_dir

if not os.path.exists(path_of_X):
  print("Can not find folder of features: "+ path_of_X +", please check and run configure.py to download or extract it!")
  sys.exit(1)

if nb_layers > 40 and nb_layers <=50:
    Maximum_length=450 # 500 will OOM  
elif nb_layers > 50:
    Maximum_length=500 
else:
    Maximum_length=450 

sample_datafile=path_of_lists + '/sample.lst'
train_datafile=path_of_lists + '/train.lst'
val_datafile=path_of_lists + '/test.lst'

import time

feature_num = load_sample_data_2D(path_of_lists, path_of_X, path_of_Y, 20000,0,dist_string, reject_fea_file)
# add error info 
print("Maximum_length % d"%Maximum_length)
start_time = time.time()

best_acc=DeepDist_train_generator(feature_num,CV_dir,net_name,out_epoch,in_epoch,rerun_epoch, filtsize,
  nb_filters,nb_layers, 1,path_of_lists,path_of_Y, path_of_X,Maximum_length,dist_string, reject_fea_file, loss_function=loss_function,
  runcount= index, if_use_binsize = False, weight = weight) #True

model_prefix = net_name
acc_history_out = "%s/%s.acc_history" % (acclog_dir, model_prefix)
chkdirs(acc_history_out)
if chkfiles(acc_history_out):
    print ('acc_file_exist,pass!')
    pass
else:
    print ('create_acc_file!')
    with open(acc_history_out, "w") as myfile:
        myfile.write("time\t netname\t filternum\t layernum\t kernelsize\t batchsize\t accuracy\n")

time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
acc_history_content = "%s\t %s\t %s\t %s\t %s\t %s\t %.4f\n" % (time_str, model_prefix, str(nb_filters),str(nb_layers),str(filtsize),str(1),best_acc)
with open(acc_history_out, "a") as myfile: myfile.write(acc_history_content) 
print("--- %s seconds ---" % (time.time() - start_time))
print("outputdir:", CV_dir)
