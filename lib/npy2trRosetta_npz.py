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

def trans_npy_into_npz(deepdist_file, npz_file, new_npz):
	mulclass = np.load(deepdist_file)
	mulclass = mulclass.squeeze()
	L = mulclass.shape[0]
	C = mulclass.shape[-1]

	P37 = mulclass[:, :, 1:37]
	P1 = mulclass[:, :, 0] + mulclass[:, :, 37:].sum(axis=-1)

	P1 = P1[:, :, np.newaxis]
	mulclass_trans = np.concatenate((P1, P37), axis=-1)

	new_npy = mulclass_trans

	npz_data = np.load(npz_file)
	dist_map = new_npy
	omega = npz_data['omega']
	theta = npz_data['theta']
	phi = npz_data['phi']
	np.savez_compressed(new_npz, dist=dist_map, omega=omega, theta=theta, phi=phi)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.description="Replace the multiclass distance map of trRosetta npz file as deepdist multiclass distance map."
    parser.add_argument("-i", "--input", help="input multiclass map",type=is_file,required=True)
    parser.add_argument("-n", "--npz", help="input trRosetta npz file",type=str,required=True)
    parser.add_argument("-o", "--output", help="output npz file",type=str,required=True)


    args = parser.parse_args()
    input_file = args.input
    npz_file = args.npz
    new_npz_file = args.output

    trans_npy_into_npz(input_file, npz_file, new_npz_file)
    