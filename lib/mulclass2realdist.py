import os,sys
import numpy as np
import argparse

def chkdirs(fn):
    '''create folder if not exists'''
    dn = os.path.dirname(fn)
    if not os.path.exists(dn): os.makedirs(dn)

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

def npy2distmap(mul_class):
    L = mul_class.shape[1]
    _class = mul_class.shape[-1]

    if _class == 38:
        mul_thred = [0, 2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5,
                    17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0]
    elif _class == 10:
        # mul_thred = [0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 18.0]
        mul_thred = [2.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]
    elif _class == 33:
        mul_thred = [1.75,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25,15.75,16.25,16.75,17.25,17.75,18.25,18.75,19.0]
    elif _class == 42:
        mul_thred = [1,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75,14.25,14.75,15.25,15.75,16.25,16.75,17.25,17.75,
        18.25,18.75,19.25,19.75,20.25,20.75,21.25,21.75,22.0]
    else:
        mul_thred = [0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.0, 15.1]

    mul_class_weighted = np.zeros((L, L, _class))
    for i in range(_class):
        mul_class_single = np.copy(mul_class[:,:,i])
        mul_class_single *= mul_thred[i]
        mul_class_weighted[:,:,i] = mul_class_single
    dist_from_mulclass = mul_class_weighted.sum(axis=-1)
    dist_from_mulclass = (dist_from_mulclass + dist_from_mulclass.T)/2.0 # this is avg of mul class
    
    return dist_from_mulclass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.description="Convert the multiclass map into distance map."
    parser.add_argument("-i", "--input", help="input multiclass map",type=is_file,required=True)
    parser.add_argument("-o", "--outdir", help="output folder",type=str,required=True)

    args = parser.parse_args()
    input_file = args.input
    outdir = args.outdir

    name = os.path.basename(input_file).split('.')[0]
    print('process %s'%name)

    npy = np.load(input_file)
    _class = npy.shape[-1]
    npy = np.squeeze(npy)
    dist = npy2distmap(npy)
    dist_file = outdir + '/' + name + '.txt'
    np.savetxt(dist_file, dist, fmt='%.4f')