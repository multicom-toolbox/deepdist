# -*- coding: utf-8 -*-
'''
Created on Wed Feb 22 21:47:26 2017

@author: Zhiye
'''
import sys
import os,glob,re
import time

sys.path.insert(0, sys.path[0])
from DNCON_lib import *
from training_strategy import *

import subprocess
import numpy as np
from keras.models import model_from_json,load_model, Sequential, Model
from keras.utils import CustomObjectScope
from random import randint
import keras.backend as K
import tensorflow as tf

from keras.engine.topology import Layer
from keras import metrics, initializers

def is_dir(dirname):
    '''Check if a path is an actual directory'''
    if not os.path.isdir(dirname):
        msg = '{0} is not a directory'.format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname

def is_file(filename):
    '''Check if a file is an invalid file'''
    if not os.path.exists(filename):
        msg = '{0} does not exist'.format(filename)
        raise argparse.ArgumentTypeError(msg)
    else:
        return filename

def chkdirs(fn):
    '''Create a folder if it doesn't already exist'''
    dn = os.path.dirname(fn)
    if not os.path.exists(dn): os.makedirs(dn)

def getFileName(path, filetype):
    f_list = os.listdir(path)
    all_file = []
    for i in f_list:
        if os.path.splitext(i)[1] == filetype:
            all_file.append(i)
    return all_file

class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

class RowNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(RowNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

class ColumNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(ColumNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

# DATABASE_FLAG
uniref90_dir ='/storage/htc/bdm/zhiye/DNCON4_db_tools//databases/uniref90_01_2020'
metaclust50_dir ='/storage/htc/bdm/zhiye/DNCON4_db_tools//databases/Metaclust_2018_06'
hhsuitedb_dir ='/storage/htc/bdm/zhiye/DNCON4_db_tools//databases/UniRef30_2020_01'
ebi_uniref100_dir ='/storage/htc/bdm/zhiye/DNCON4_db_tools//databases/myg_uniref100_01_2020'
# End of configure

if len(sys.argv) == 11:
    db_tool_dir = os.path.abspath(sys.argv[1])
    fasta = os.path.abspath(sys.argv[2])
    CV_dir = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]] # ensemble use four model average
    outdir = os.path.abspath(sys.argv[7])
    predict_method = str(sys.argv[8])  # bin_class real_dist_limited16 real_dist_scaled mul_class dist_error real_dist
    option = str(sys.argv[9])
    serve_name = str(sys.argv[10])
elif len(sys.argv) == 8:
    db_tool_dir = os.path.abspath(sys.argv[1])
    fasta = os.path.abspath(sys.argv[2])
    CV_dir = [sys.argv[3]]
    outdir = os.path.abspath(sys.argv[4])
    predict_method = str(sys.argv[5])
    option = str(sys.argv[6])
    serve_name = str(sys.argv[7])
else:
    print('Please input the right parameters:\n')
    print('[db_tool_dir] [fasta_file] [model_dir] [output_dir] [predict_method]')
    sys.exit(1)

print(f'Model directory: {CV_dir}')
print(f'Predict method: {predict_method}')
only_predict_flag = True # if do not have label set True
lib_path = sys.path[0]
global_path = os.path.dirname(sys.path[0])
print(f'Global path: {global_path}')
path_of_X = outdir
path_of_Y = outdir

feature_list = 'other' # ['combine', 'combine_all2d', 'other', 'ensemble']  # 'combine' will output three map and combines them, 'other' outputs one pred
data_list_choose = 'test' # ['train', 'test', 'train_sub', 'all']
maximum_length = 2000  # casp12 700
dist_string = '80'
loss_function = 'binary_crossentropy'
if_use_binsize = False # False True
save_mul_real = True

db_tool_dir = os.path.abspath(sys.argv[1])
script_path = global_path+'/scripts/'
target = os.path.basename(fasta)
target = re.sub('\.fasta','',target)

##########################################################
# Generating four set of features and alignment
##########################################################
if not os.path.exists(fasta):
    print(f'Cannot find fasta file: {fasta}')
    sys.exit(1)

if not os.path.exists(outdir):
    os.makedirs(outdir)
    print(f'Create output folder path: {outdir}')
if os.path.exists(outdir+'/X-'+target+'.txt') and os.path.exists(outdir+'/'+target+'.cov') and os.path.exists(outdir+'/'+target+'.plm') and os.path.exists(outdir+'/'+target+'.pre'):
    print('Features already exist... skip')
else:
    # Step 1: Generate alignment
    if os.path.exists(outdir+'/alignment/'+target+'.aln') and os.path.getsize(outdir+'/alignment/'+target+'.aln') > 0:
        print('Alignment already generated... skip')
    else:
        if hhsuitedb_dir.split('/')[-1] =='':
            hhsuitedb_name = hhsuitedb_dir.split('/')[-2]
        else:
            hhsuitedb_name = hhsuitedb_dir.split('/')[-1]
        if option == 'ALN':
            os.system(db_tool_dir+'/tools/DeepAlign1.0/hhjack_hhmsearch3.sh '+fasta+' '+outdir+'/alignment '+db_tool_dir+'/tools/ '+db_tool_dir+'/databases/'+' '+uniref90_dir+'/uniref90 '+hhsuitedb_dir+'/'+hhsuitedb_name+' '+metaclust50_dir+'/metaclust_50')
        elif option == 'MSA':
            os.system('python '+db_tool_dir+'/tools/deepmsa/hhsuite2/scripts/build_MSA.py '+fasta+' -hhblitsdb='+hhsuitedb_dir+'/'+hhsuitedb_name+' -jackhmmerdb='
                +uniref90_dir+'/uniref90 -hmmsearchdb='+ebi_uniref100_dir+'/myg_uniref100 -tmpdir='+outdir+'/alignment/tmp -outdir='
                +outdir+'/alignment -ncpu=8 -overwrite=0')
        else:
            print('Default set DeepAln pipline!\n')
            os.system(db_tool_dir+'/tools/DeepAlign1.0/hhjack_hhmsearch3.sh '+fasta+' '+outdir+'/alignment '+db_tool_dir+'/tools/ '+db_tool_dir+'/databases/'+' '+uniref90_dir+'/uniref90 '+hhsuitedb_dir+'/'+hhsuitedb_name+' '+metaclust50_dir+'/metaclust_50')
        if os.path.exists(outdir+'/alignment/'+target+'.aln') and os.path.getsize(outdir+'/alignment/'+target+'.aln') > 0:
            print('Alignment generated successfully')
        else:
            print('Alignment generation failed')

    # Step 2: Generate OTHER
    if os.path.exists(outdir+'/X-'+target+'.txt') and os.path.getsize(outdir+'/X-'+target+'.txt') > 0:
        print('DNCON2 features already generated... skip')
    else:
        os.system('perl '+script_path+'/generate-other.pl '+db_tool_dir+' '+fasta+' '+outdir+' '+uniref90_dir+'/uniref90')
        if os.path.exists(outdir+'/X-'+target+'.txt') and os.path.getsize(outdir+'/X-'+target+'.txt') > 0:
            print('DNCON2 features generated successfully')
        else:
            print('DNCON2 feature generation failed')

    # Step 3: Generate COV
    if os.path.exists(outdir+'/'+target+'.cov') and os.path.getsize(outdir+'/'+target+'.cov') > 0:
        print('COV already generated... skip')
    else:
        os.system(script_path+'/cov21stats '+outdir+'/alignment/'+target+'.aln '+outdir+'/'+target+'.cov')
        if os.path.exists(outdir+'/'+target+'.cov') and os.path.getsize(outdir+'/'+target+'.cov') > 0:
            print('COV generated successfully')
        else:
            print('COV generation failed')

    # Step 4: Generate PLM
    if os.path.exists(outdir+'/ccmpred/'+target+'.plm') and os.path.getsize(outdir+'/ccmpred/'+target+'.plm') > 0:
        print('PLM already generated... skip')
        os.system('mv '+outdir+'/ccmpred/'+target+'.plm '+outdir)
    elif os.path.exists(outdir+'/'+target+'.plm') and os.path.getsize(outdir+'/'+target+'.plm') > 0:
        print('PLM already generated... skip')
    else:
        print('PLM generation failed')

    # Step 5: Generate PRE
    if os.path.exists(outdir+'/'+target+'.pre') and os.path.getsize(outdir+'/'+target+'.pre') > 0:
        print('PRE already generated... skip')
    else:
        os.system(script_path+'/calNf_ly '+outdir+'/alignment/'+target+'.aln 0.8 > '+outdir+'/'+target+'.weight')
        os.system('python -W ignore '+script_path+'/generate_pre.py '+outdir+'/alignment/'+target+'.aln '+outdir+'/'+target)
        os.system('rm '+outdir+'/'+target+'.weight')
        if os.path.exists(outdir+'/'+target+'.pre') and os.path.getsize(outdir+'/'+target+'.pre') > 0:
            print('PRE generated successfully')
        else:
            print('PRE generation failed')

# gpu_schedul_strategy('local', allow_growth=True)

print('\n######################################\n佛祖保佑，永不迨机，永无bug，精度九十九\n######################################\n')

length = 0
f = open(fasta, 'r')
for line in f.readlines():
    if line.startswith('>'):
        continue
    else:
        length = len(line.strip('\n'))
if length == 0:
    print(f'Read fasta: {fasta} length wrong!')
selected_list = {}
selected_list[target] = length

print(f'Total number to predict: {len(selected_list)}')

iter_num = 0
if isinstance(CV_dir, str) == True:
    iter_num = 1
    CV_dir = [CV_dir]
else:
    iter_num = len(CV_dir)
chkdirs(outdir)

for index in range(iter_num):
    sub_cv_dir = CV_dir[index]
    reject_fea_path = sub_cv_dir + '/'
    reject_fea_file = getFileName(reject_fea_path, '.txt')

    model_out= sub_cv_dir + '/' + getFileName(sub_cv_dir, '.json')[0]
    model_weight_out_best = sub_cv_dir + '/' + getFileName(sub_cv_dir, '.h5')[0]
    model_weight_top10 = f'{sub_cv_dir}/model_weights_top/'

    # pred_history_out = '%s/predict%d.acc_history' % (outdir, index) 
    # with open(pred_history_out, 'a') as myfile:
    #     myfile.write(time.strftime('%Y-%m-%d %H:%M:%S\n',time.localtime(time.time())))
    with CustomObjectScope({'InstanceNormalization': InstanceNormalization, 'RowNormalization': RowNormalization, 'ColumNormalization': ColumNormalization, 'tf':tf}):
        json_string = open(model_out).read()
        DNCON4 = model_from_json(json_string)

    if os.path.exists(model_weight_out_best):
        print(f'Loading existing weights: {model_weight_out_best}')
        DNCON4.load_weights(model_weight_out_best)
    else:
        print('Please check the best weights')

    if serve_name == None or serve_name == 'None':
        preddir = outdir
    else:
        preddir = outdir + '/' + serve_name

    if 'mul_class' in predict_method:
        model_predict= f'{preddir}/pred_map{index}/'
        chkdirs(model_predict)
        mul_class_dir= f'{model_predict}/mul_class/'
        chkdirs(mul_class_dir)
    elif 'real_dist' in predict_method: 
        model_predict= f'{preddir}/pred_map{index}/'
        chkdirs(model_predict)
        real_dist_dir= f'{model_predict}/real_dist/'
        chkdirs(real_dist_dir)
    elif 'mul_lable' in  predict_method:
        real_dist_bin_dir = f'{preddir}/pred_map_real_dist_{index}/'
        mul_class_bin_dir = f'{preddir}/pred_map_mul_class_{index}/'
        real_dist_dir= f'{real_dist_bin_dir}/real_dist/'
        mul_class_dir = f'{mul_class_bin_dir}/mul_class/'
        model_predict = real_dist_bin_dir
        chkdirs(real_dist_bin_dir)
        chkdirs(mul_class_bin_dir)
        chkdirs(real_dist_dir)
        chkdirs(mul_class_dir)

    if 'other' == feature_list:
        if len(reject_fea_file) == 1:
            OTHER = reject_fea_path + reject_fea_file[0]
            # print(OTHER)
        elif len(reject_fea_file) >= 2:
            OTHER = []
            for feafile_num in range(len(reject_fea_file)):
                OTHER.append(reject_fea_path + reject_fea_file[feafile_num])

    for key in selected_list:
        value = selected_list[key]
        p1 = {key: value}
        if if_use_binsize:
            maximum_length = maximum_length
        else:
            maximum_length = value
        if len(p1) < 1:
            continue
        print(f'Predicting {key} {value}\n')

        if 'other' in feature_list:
            if len(reject_fea_file) == 1:
                selected_list_2D_other = get_x_2D_from_this_list_pred(p1, path_of_X, maximum_length,dist_string, OTHER, value)
                if type(selected_list_2D_other) == bool:
                    continue
                DNCON4_prediction_other = DNCON4.predict([selected_list_2D_other], batch_size= 1)  
            elif len(reject_fea_file)>=2:
                pred_temp = []
                bool_flag = False
                for fea_num in range(len(OTHER)):
                    temp = get_x_2D_from_this_list_pred(p1, path_of_X, maximum_length, dist_string, reject_fea_file[fea_num], value)
                    # print('selected_list_2D.shape: ',temp.shape)
                    if type(temp) == bool:
                        bool_flag= True
                    pred_temp.append(temp)
                if bool_flag == True:
                    continue
                else:
                    DNCON4_prediction_other = DNCON4.predict(pred_temp, batch_size= 1)
            
            if predict_method == 'mul_class':
                DNCON4_prediction_dist = np.copy(DNCON4_prediction_other)
                DNCON4_prediction_other= DNCON4_prediction_other[:,:,:,0:8].sum(axis=-1)
            elif predict_method == 'mul_class_C':
                DNCON4_prediction_dist = np.copy(DNCON4_prediction_other)
                DNCON4_prediction_other= DNCON4_prediction_other[:,:,:,0:3].sum(axis=-1)
            elif predict_method == 'mul_class_D':
                DNCON4_prediction_dist = np.copy(DNCON4_prediction_other)
                DNCON4_prediction_other= DNCON4_prediction_other[:,:,:,0:10].sum(axis=-1)
            elif predict_method == 'mul_class_T':
                DNCON4_prediction_dist = np.copy(DNCON4_prediction_other)
                DNCON4_prediction_other= DNCON4_prediction_other[:,:,:,0:13].sum(axis=-1)
            elif predict_method == 'mul_class_G':
                DNCON4_prediction_dist = np.copy(DNCON4_prediction_other)
                DNCON4_prediction_other= DNCON4_prediction_other[:,:,:,0:13].sum(axis=-1)
            elif predict_method == 'real_dist':
                DNCON4_prediction_other[DNCON4_prediction_other>100] = 100 # incase infinity
                DNCON4_prediction_other[DNCON4_prediction_other<=0] = 1 # incase infinity
                DNCON4_prediction_dist = np.copy(DNCON4_prediction_other)
                DNCON4_prediction_other = 1/DNCON4_prediction_other # convert to confidence
            elif predict_method == 'mul_lable':
                mul_class = DNCON4_prediction_other[0]
                DNCON4_prediction_mul_class= DNCON4_prediction_other[0][:,:,:,0:8].sum(axis=-1)
                DNCON4_prediction_real_dist= DNCON4_prediction_other[1]
                DNCON4_prediction_real_dist[DNCON4_prediction_real_dist>100] = 100 
                DNCON4_prediction_real_dist[DNCON4_prediction_real_dist<=0] = 1
                DNCON4_prediction_dist = np.copy(DNCON4_prediction_real_dist)
                DNCON4_prediction_dist = 1/DNCON4_prediction_dist
                # DNCON4_prediction_other = (DNCON4_prediction_mul_class.reshape(maximum_length, maximum_length) + DNCON4_prediction_dist.reshape(maximum_length, maximum_length))/2.0
                real_dist_bin = (DNCON4_prediction_dist.reshape(maximum_length, maximum_length))
                mul_class_bin = (DNCON4_prediction_mul_class.reshape(maximum_length, maximum_length))
                Map_UpTrans = (np.triu(real_dist_bin, 1).T + np.tril(real_dist_bin, -1))/2
                Map_UandL = (np.triu(real_dist_bin) + np.tril(real_dist_bin).T)/2
                real_dist_bin = Map_UandL + Map_UpTrans
                Map_UpTrans = (np.triu(mul_class_bin, 1).T + np.tril(mul_class_bin, -1))/2
                Map_UandL = (np.triu(mul_class_bin) + np.tril(mul_class_bin).T)/2
                mul_class_bin = Map_UandL + Map_UpTrans
                DNCON4_prediction_other = (DNCON4_prediction_dist.reshape(maximum_length, maximum_length))
                DNCON4_prediction_dist = DNCON4_prediction_real_dist
            elif predict_method == 'mul_lable_R':
                mul_class = DNCON4_prediction_other[0]
                DNCON4_prediction_mul_class= DNCON4_prediction_other[0][:,:,:,0:8].sum(axis=-1)
                DNCON4_prediction_real_dist= DNCON4_prediction_other[1]
                DNCON4_prediction_real_dist[DNCON4_prediction_real_dist>100] = 100 
                DNCON4_prediction_real_dist[DNCON4_prediction_real_dist<=0] = 1
                DNCON4_prediction_dist = np.copy(DNCON4_prediction_real_dist)
                DNCON4_prediction_dist = 1/DNCON4_prediction_dist
                # DNCON4_prediction_other = (DNCON4_prediction_mul_class.reshape(maximum_length, maximum_length) + DNCON4_prediction_dist.reshape(maximum_length, maximum_length))/2.0
                real_dist_bin = (DNCON4_prediction_dist.reshape(maximum_length, maximum_length))
                mul_class_bin = (DNCON4_prediction_mul_class.reshape(maximum_length, maximum_length))
                Map_UpTrans = (np.triu(real_dist_bin, 1).T + np.tril(real_dist_bin, -1))/2
                Map_UandL = (np.triu(real_dist_bin) + np.tril(real_dist_bin).T)/2
                real_dist_bin = Map_UandL + Map_UpTrans
                Map_UpTrans = (np.triu(mul_class_bin, 1).T + np.tril(mul_class_bin, -1))/2
                Map_UandL = (np.triu(mul_class_bin) + np.tril(mul_class_bin).T)/2
                mul_class_bin = Map_UandL + Map_UpTrans
                DNCON4_prediction_other = (DNCON4_prediction_dist.reshape(maximum_length, maximum_length))
                DNCON4_prediction_dist = DNCON4_prediction_real_dist

            CMAP = DNCON4_prediction_other.reshape(maximum_length, maximum_length)
            Map_UpTrans = (np.triu(CMAP, 1).T + np.tril(CMAP, -1))/2
            Map_UandL = (np.triu(CMAP) + np.tril(CMAP).T)/2
            real_cmap_other = Map_UandL + Map_UpTrans
            other_cmap_file = f'{model_predict}/{key}.txt'
            np.savetxt(other_cmap_file, real_cmap_other, fmt='%.4f')
            # real_cmap_other = CMAP
            if 'mul_lable' in predict_method and save_mul_real == True:
                CMAP = DNCON4_prediction_dist.reshape(maximum_length, maximum_length)
                real_dmap_dist = (CMAP + CMAP.T)/2
                real_dmap_file = f'{real_dist_dir}/{key}.txt'
                realdist_cmap_file = f'{real_dist_bin_dir}/{key}.txt'
                mulclass_cmap_file = f'{mul_class_bin_dir}/{key}.txt'
                mulclass_file = f'{mul_class_dir}/{key}.npy'
                np.savetxt(real_dmap_file, real_dmap_dist, fmt='%.4f')
                np.savetxt(realdist_cmap_file, real_dist_bin, fmt='%.4f')
                np.savetxt(mulclass_cmap_file, mul_class_bin, fmt='%.4f')
                np.save(mulclass_file, mul_class)
            elif 'mul_class' in predict_method:
                CMAP = DNCON4_prediction_dist.reshape(1, maximum_length, maximum_length, -1)
                real_dmap_dist = (CMAP + CMAP.transpose(0,2,1,3))/2
                other_dmap_file = f'{mul_class_dir}/{key}.npy'
                np.save(other_dmap_file, real_dmap_dist)
            elif predict_method == 'dist_error':
                CMAP = DNCON4_prediction_error.reshape(maximum_length, maximum_length)
                dist_error_map = (CMAP+CMAP.T)/2
                error_file = f'{real_dist_dir}/{key}.error'
                np.savetxt(error_file, dist_error_map, fmt='%.4f')
            elif predict_method == 'real_dist':
                CMAP = DNCON4_prediction_dist.reshape(maximum_length, maximum_length)
                real_dmap_dist = (CMAP + CMAP.T)/2
                real_dmap_file = f'{real_dist_dir}/{key}.txt'
                np.savetxt(real_dmap_file, real_dmap_dist, fmt='%.4f')

##########################################################
# Using CONEVA to evaluate
##########################################################
if iter_num == 1: # This is the single model predictor
    if 'mul_lable' not in predict_method:
        cmap_dir= f'{preddir}/pred_map{index}/'
    else:
        cmap_dir= f'{preddir}/pred_map_real_dist_{index}/'

    rr_dir = cmap_dir+'/rr/'
    chkdirs(rr_dir)
    os.chdir(rr_dir)
    for filename in glob.glob(cmap_dir+'/*.txt'):
        id = os.path.basename(filename)
        id = re.sub('\.txt$', '', id)
        f = open(rr_dir+'/'+id+'.raw','w')
        cmap = np.loadtxt(filename,dtype='float32')
        L = cmap.shape[0]
        for i in range(0,L):
            for j in range(i+1,L):
                f.write(str(i+1)+' '+str(j+1)+' 0 8 '+str(cmap[i][j])+'\n')
        f.close()
        os.system('egrep -v \'^>\' '+ fasta +'  > '+id+'.rr')
        os.system('cat '+id+'.raw >> '+id+'.rr')
        os.system('rm -f '+id+'.raw')
    if only_predict_flag == False:
        print('Running CONEVA... May take 1 or 2 minutes\n')
        emoji_flag = False
        for key in selected_list:
            # print(key+' evaluated')print
            if emoji_flag:
                emoji_flag=False
                print('\r', '\\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))', end='', flush=True)
            else:
                emoji_flag=True
                print('\r', ' ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/', end='', flush=True)
            pdb_name = get_all_file_contain_str(path_of_pdb, key)
            for i in range(len(pdb_name)):
                pdb_file = path_of_pdb + pdb_name[i]
                if os.path.exists(pdb_file):
                    subprocess.call('perl '+lib_path+'/coneva-lite.pl -rr '+rr_dir+'/'+key+'.rr -pdb '+ pdb_file + ' >> '+rr_dir+'/rr.txt',shell=True)
                else:
                    print(f'Please check the pdb file: {pdb_file}')
        title_line = '\nPRECISION                     Top-5     Top-L/10  Top-L/5   Top-L/2   Top-L     Top-2L    '
        with open(final_acc_reprot, 'a') as myfile:
            myfile.write(title_line)
            myfile.write('\n')
        print(title_line)
        
        top5_acc = topL10_acc = topL5_acc = topL2_acc = topL_acc = top2L_acc = 0 
        count = 0
        for line in open(rr_dir+'/rr.txt', 'r'):
            line = line.rstrip()
            if('.pdb (precision)' in line):
                arr = line.split()
                print(arr[0])  
                with open(final_acc_reprot, 'a') as myfile:
                    myfile.write(arr[0])
                    myfile.write('\n')
            if('.rr (precision)' in line):
                count += 1
                print(line, end=' ')
                with open(final_acc_reprot, 'a') as myfile:
                    myfile.write(line)
                _array = line.split(' ')
                array = [x for x in _array if x !='']
                top5_acc   += float(array[2])
                topL10_acc += float(array[3])
                topL5_acc  += float(array[4])
                topL2_acc  += float(array[5])
                topL_acc   += float(array[6])
                top2L_acc  += float(array[7])
        top5_acc   /= count
        topL10_acc /= count
        topL5_acc  /= count
        topL2_acc  /= count
        topL_acc   /= count
        top2L_acc  /= count
        final_line = f'AVERAGE                       {top5_acc:.2f}     {topL10_acc:.2f}    {topL5_acc:.2f}     {topL2_acc:.2f}     {topL_acc:.2f}     {top2L_acc:.2f}    \n'
        print(final_line)
        with open(final_acc_reprot, 'a') as myfile:
            myfile.write(final_line)
        os.system('rm -f rr.txt')
    else:
        print(f'Final pred_map filepath: {cmap_dir}')
        print(f'Final rr       filepath: {rr_dir}')
elif iter_num == 4: # This is the multiple model predictor, now with 4 models
    if 'mul_lable' not in predict_method:
        cmap1dir = f'{preddir}/pred_map0/'
        cmap2dir = f'{preddir}/pred_map1/'
        cmap3dir = f'{preddir}/pred_map2/'
        cmap4dir = f'{preddir}/pred_map3/'
    else:
        cmap1dir = f'{preddir}/pred_map_real_dist_0/'
        cmap2dir = f'{preddir}/pred_map_real_dist_1/'
        cmap3dir = f'{preddir}/pred_map_real_dist_2/'
        cmap4dir = f'{preddir}/pred_map_real_dist_3/'
    sum_cmap_dir = f'{preddir}/pred_map_ensem/'
    sum_real_dir = f'{sum_cmap_dir}/real_dist/'
    chkdirs(sum_cmap_dir)
    chkdirs(sum_real_dir)
    for key in selected_list:
        seq_name = key
        print(f'Processing {seq_name}')
        sum_map_filename = sum_cmap_dir + seq_name + '.txt'
        real_dist_filename = sum_real_dir + seq_name + '.txt'
        cmap1 = np.loadtxt(cmap1dir + seq_name + '.txt', dtype=np.float32)
        cmap2 = np.loadtxt(cmap2dir + seq_name + '.txt', dtype=np.float32)
        cmap3 = np.loadtxt(cmap3dir + seq_name + '.txt', dtype=np.float32)
        cmap4 = np.loadtxt(cmap4dir + seq_name + '.txt', dtype=np.float32)
        sum_map = (cmap1 * 0.22 + cmap2 * 0.34 + cmap3 * 0.22 + cmap4 * 0.22)
        real_dist = 1/(sum_map+1e-10)
        np.savetxt(sum_map_filename, sum_map, fmt='%.4f')
        np.savetxt(real_dist_filename, real_dist, fmt='%.4f')
    if 'mul_class' in predict_method:
        npy1dir = f'{preddir}/pred_map0/mul_class/'
        npy2dir = f'{preddir}/pred_map1/mul_class/'
        npy3dir = f'{preddir}/pred_map2/mul_class/'
        npy4dir = f'{preddir}/pred_map3/mul_class/'
        sum_npy_dir = f'{preddir}/pred_map_ensem/mul_class/'
        chkdirs(sum_npy_dir)        
        for key in selected_list:
            seq_name = key
            sum_npy_filename = sum_npy_dir + seq_name + '.npy'
            npy1 = np.load(npy1dir + seq_name + '.npy')
            npy2 = np.load(npy2dir + seq_name + '.npy')
            npy3 = np.load(npy3dir + seq_name + '.npy')
            npy4 = np.load(npy4dir + seq_name + '.npy')
            sum_npy = (npy1 * 0.22 + npy2 * 0.34 + npy3 * 0.22 + npy4 * 0.22)
            np.save(sum_npy_filename, sum_npy)

    cmap_dir= sum_cmap_dir
    rr_dir = cmap_dir+'/rr/'
    chkdirs(rr_dir)
    os.chdir(rr_dir)
    for filename in glob.glob(cmap_dir+'/*.txt'):
        id = os.path.basename(filename)
        id = re.sub('\.txt$', '', id)
        f = open(rr_dir+'/'+id+'.raw','w')
        cmap = np.loadtxt(filename,dtype='float32')
        L = cmap.shape[0]
        for i in range(0,L):
            for j in range(i+1,L):
                f.write(str(i+1)+' '+str(j+1)+' 0 8 '+str(cmap[i][j])+'\n')
        f.close()
        os.system('egrep -v \'^>\' '+ fasta +'  > '+id+'.rr')
        os.system('cat '+id+'.raw >> '+id+'.rr')
        os.system('rm -f '+id+'.raw')
    if only_predict_flag == False:
        print('Running CONEVA... May take 1 or 2 minutes\n')
        emoji_flag = False
        for key in selected_list:
            # print(key+' evaluated')print
            if emoji_flag:
                emoji_flag=False
                print('\r', '\\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))', end='', flush=True)
            else:
                emoji_flag=True
                print('\r', ' ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/', end='', flush=True)
            pdb_name = get_all_file_contain_str(path_of_pdb, key)
            for i in range(len(pdb_name)):
                pdb_file = path_of_pdb + pdb_name[i]
                if os.path.exists(pdb_file):
                    subprocess.call('perl '+lib_path+'/coneva-lite.pl -rr '+rr_dir+'/'+key+'.rr -pdb '+ pdb_file + ' >> '+rr_dir+'/rr.txt',shell=True)
                else:
                    print(f'Please check the pdb file: {pdb_file}')
        title_line = '\nPRECISION                     Top-5     Top-L/10  Top-L/5   Top-L/2   Top-L     Top-2L    '
        with open(final_acc_reprot, 'a') as myfile:
            myfile.write(title_line)
            myfile.write('\n')
        print(title_line)
        
        top5_acc = topL10_acc = topL5_acc = topL2_acc = topL_acc = top2L_acc = 0 
        count = 0
        for line in open(rr_dir+'/rr.txt','r'):
            line = line.rstrip()
            if('.pdb (precision)' in line):
                arr = line.split()
                print(arr[0])  
                with open(final_acc_reprot, 'a') as myfile:
                    myfile.write(arr[0])
                    myfile.write('\n')
            if('.rr (precision)' in line):
                count += 1
                print(line, end=' ')
                with open(final_acc_reprot, 'a') as myfile:
                    myfile.write(line)
                _array = line.split(' ')
                array = [x for x in _array if x !='']
                top5_acc   += float(array[2])
                topL10_acc += float(array[3])
                topL5_acc  += float(array[4])
                topL2_acc  += float(array[5])
                topL_acc   += float(array[6])
                top2L_acc  += float(array[7])
        top5_acc   /= count
        topL10_acc /= count
        topL5_acc  /= count
        topL2_acc  /= count
        topL_acc   /= count
        top2L_acc  /= count
        final_line = f'AVERAGE                       {top5_acc:.2f}     {topL10_acc:.2f}     {topL5_acc:.2f}     {topL2_acc:.2f}     {topL_acc:.2f}     {top2L_acc:.2f}    \n'
        print(final_line)
        with open(final_acc_reprot, 'a') as myfile:
            myfile.write(final_line)
        os.system('rm -f rr.txt')                    
    else:
        print(f'Final pred_map filepath: {cmap_dir}')
        print(f'Final rr       filepath: {rr_dir}')
    if 'mul_lable' in predict_method:
        cmap1dir = f'{preddir}/pred_map_mul_class_0/'
        cmap2dir = f'{preddir}/pred_map_mul_class_1/'
        cmap3dir = f'{preddir}/pred_map_mul_class_2/'
        cmap4dir = f'{preddir}/pred_map_mul_class_3/'
        sum_cmap_dir = f'{preddir}/pred_map_mul_class_ensem/'

        npy1dir = f'{preddir}/pred_map_mul_class_0/mul_class/'
        npy2dir = f'{preddir}/pred_map_mul_class_1/mul_class/'
        npy3dir = f'{preddir}/pred_map_mul_class_2/mul_class/'
        npy4dir = f'{preddir}/pred_map_mul_class_3/mul_class/'
        sum_npy_dir = f'{preddir}/pred_map_mul_class_ensem/mul_class/'
        chkdirs(sum_cmap_dir)
        chkdirs(sum_npy_dir)
        for key in selected_list:
            seq_name = key
            print(f'Processing {seq_name}')
            sum_map_filename = sum_cmap_dir + seq_name + '.txt'
            cmap1 = np.loadtxt(cmap1dir + seq_name + '.txt', dtype=np.float32)
            cmap2 = np.loadtxt(cmap2dir + seq_name + '.txt', dtype=np.float32)
            cmap3 = np.loadtxt(cmap3dir + seq_name + '.txt', dtype=np.float32)
            cmap4 = np.loadtxt(cmap4dir + seq_name + '.txt', dtype=np.float32)
            sum_map = (cmap1 * 0.22 + cmap2 * 0.34 + cmap3 * 0.22 + cmap4 * 0.22)
            np.savetxt(sum_map_filename, sum_map, fmt='%.4f')

            sum_npy_filename = sum_npy_dir + seq_name + '.npy'
            npy1 = np.load(npy1dir + seq_name + '.npy')
            npy2 = np.load(npy2dir + seq_name + '.npy')
            npy3 = np.load(npy3dir + seq_name + '.npy')
            npy4 = np.load(npy4dir + seq_name + '.npy')
            sum_npy = (npy1 * 0.22 + npy2 * 0.34 + npy3 * 0.22 + npy4 * 0.22)
            np.save(sum_npy_filename, sum_npy)
        
        cmap_dir= sum_cmap_dir
        rr_dir = cmap_dir+'/rr/'
        chkdirs(rr_dir)
        os.chdir(rr_dir)
        for filename in glob.glob(cmap_dir+'/*.txt'):
            id = os.path.basename(filename)
            id = re.sub('\.txt$', '', id)
            f = open(rr_dir+'/'+id+'.raw','w')
            cmap = np.loadtxt(filename,dtype='float32')
            L = cmap.shape[0]
            for i in range(0,L):
                for j in range(i+1,L):
                    f.write(str(i+1)+' '+str(j+1)+' 0 8 '+str(cmap[i][j])+'\n')
            f.close()
            os.system('egrep -v \'^>\' '+ fasta +'  > '+id+'.rr')
            os.system('cat '+id+'.raw >> '+id+'.rr')
            os.system('rm -f '+id+'.raw')
        if only_predict_flag == False:
            print('Running CONEVA... May take 1 or 2 minutes\n')
            emoji_flag = False
            for key in selected_list:
                # print(key+' evaluated')print
                if emoji_flag:
                    emoji_flag=False
                    print('\r', '\\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))  \\(￣︶￣*\\))', end='', flush=True)
                else:
                    emoji_flag=True
                    print('\r', ' ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/  ((/*￣︶￣)/', end='', flush=True)
                pdb_name = get_all_file_contain_str(path_of_pdb, key)
                for i in range(len(pdb_name)):
                    pdb_file = path_of_pdb + pdb_name[i]
                    if os.path.exists(pdb_file):
                        subprocess.call('perl '+lib_path+'/coneva-lite.pl -rr '+rr_dir+'/'+key+'.rr -pdb '+ pdb_file + ' >> '+rr_dir+'/rr.txt',shell=True)
                    else:
                        print(f'Please check the pdb file: {pdb_file}')
            title_line = '\nPRECISION                     Top-5     Top-L/10  Top-L/5   Top-L/2   Top-L     Top-2L    '
            with open(final_acc_reprot, 'a') as myfile:
                myfile.write(title_line)
                myfile.write('\n')
            print(title_line)
            
            top5_acc = topL10_acc = topL5_acc = topL2_acc = topL_acc = top2L_acc = 0 
            count = 0
            for line in open(rr_dir+'/rr.txt','r'):
                line = line.rstrip()
                if('.pdb (precision)' in line):
                    arr = line.split()
                    print(arr[0])  
                    with open(final_acc_reprot, 'a') as myfile:
                        myfile.write(arr[0])
                        myfile.write('\n')
                if('.rr (precision)' in line):
                    count += 1
                    print(line, end=' ')
                    with open(final_acc_reprot, 'a') as myfile:
                        myfile.write(line)
                    _array = line.split(' ')
                    array = [x for x in _array if x !='']
                    top5_acc   += float(array[2])
                    topL10_acc += float(array[3])
                    topL5_acc  += float(array[4])
                    topL2_acc  += float(array[5])
                    topL_acc   += float(array[6])
                    top2L_acc  += float(array[7])
            top5_acc   /= count
            topL10_acc /= count
            topL5_acc  /= count
            topL2_acc  /= count
            topL_acc   /= count
            top2L_acc  /= count
            final_line = f'AVERAGE                       {top5_acc:.2f}     {topL10_acc:.2f}     {topL5_acc:.2f}     {topL2_acc:.2f}     {topL_acc:.2f}     {top2L_acc:.2f}    \n'
            print(final_line)
            with open(final_acc_reprot, 'a') as myfile:
                myfile.write(final_line)
            os.system('rm -f rr.txt')                                           
        else:
            print(f'Final pred_map filepath: {cmap_dir}')
            print(f'Final rr       filepath: {rr_dir}')

print('DeepDist has finished running\n')