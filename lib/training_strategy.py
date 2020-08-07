import os
import sys
import numpy as np
import platform
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def gpu_schedul_strategy(sysflag, gpu_mem_rate = 0.5, allow_growth = False):
    gpu_mem = 0
    current_path = sys.path[0]
    if sysflag == 'local':
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
        os.system('chmod -R 777 tmp')
        memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
        if memory_gpu == []:
            print("System is out of GPU memory, Run on CPU")
            os.environ['CUDA_VISIBLE_DEVICES']="-1"
        else:
            gpu_mem = np.max(memory_gpu)
            if np.max(memory_gpu) <= 2000:
                print("System is out of GPU memory, Run on CPU")
                os.environ['CUDA_VISIBLE_DEVICES']="-1"
                os.system('rm tmp')
                sys.exit(1)
            else:
                os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
                os.system('rm tmp')
        print("Run on GPU %s\n" %(str(np.argmax(memory_gpu))))
    else:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
        os.system('chmod -R 777 tmp')
        memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
        if memory_gpu == []:
            print("System is out of GPU memory, Run on CPU")
            os.environ['CUDA_VISIBLE_DEVICES']="-1"
        else:
            gpu_mem = np.max(memory_gpu)
            if np.max(memory_gpu) <= 10000:
                print("System is out of GPU memory, Run on CPU")
                os.environ['CUDA_VISIBLE_DEVICES']="-1"
                os.system('rm tmp')
                sys.exit(1)
            else:
                os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
                os.system('rm tmp')
        print("Run on GPU %s\n" %(str(np.argmax(memory_gpu))))
    config = tf.ConfigProto()
    if (allow_growth):
        config.gpu_options.allow_growth = allow_growth
    else:
        config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_rate
        sess= tf.Session(config = config)
        KTF.set_session(sess)
    return gpu_mem

if __name__ == '__main__':
    current_os_name = platform.platform()
    print('%s' % current_os_name)
    if 'Ubuntu' in current_os_name.split('-'): #on local
      sysflag='local'
    elif 'centos' in current_os_name.split('-'): #on lewis or multicom
      sysflag='lewis'
    print(gpu_schedul_strategy(sysflag))