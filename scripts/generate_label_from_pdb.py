import glob,re,os,sys
import math
import argparse
import numpy as np

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

#G: 0-2,   2-22:0.5,   22-∞
#C: 0-4,   4-20:2,     20-∞
#D: 0-3.5, 3.5-19:0.5, 20-∞
#T: 0-2,   2-20:0.5,   20-∞
def real_value2mul_class(input_mat, option='G'):
    length = input_mat.shape[0]
    output_mat = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if option == 'G':
                output_mat[i,j] = math.ceil(input_mat[i,j]/0.5 - 4)
                if output_mat[i,j] < -3:
                    output_mat[i, j] = -1 #gap in the map
                elif output_mat[i,j] < 0:
                    output_mat[i, j] = 0
                elif output_mat[i,j] > 41:
                    output_mat[i, j] = 41
            elif option == 'C':
                output_mat[i,j] = math.ceil(input_mat[i,j]/2 - 2)
                if output_mat[i,j] < -1:
                    output_mat[i, j] = -1
                elif output_mat[i,j] < 0:
                    output_mat[i, j] = 0
                elif output_mat[i,j] > 9:
                    output_mat[i, j] = 9
            elif option == 'D':
                output_mat[i,j] = math.ceil(input_mat[i,j]/0.5 - 7)
                if output_mat[i,j] < -6:
                    output_mat[i, j] = -1
                elif output_mat[i,j] < 0:
                    output_mat[i, j] = 0
                elif output_mat[i,j] > 32:
                    output_mat[i, j] = 32
            elif option == 'T':
                output_mat[i,j] = math.ceil(input_mat[i,j]/0.5 - 4) 
                if output_mat[i,j] < -3:
                    output_mat[i, j] = -1
                elif output_mat[i,j] < 0:
                    output_mat[i, j] = 0
                elif output_mat[i,j] > 36:
                    output_mat[i, j] = 36
    return output_mat

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.description="Generate real value distance map, multi-class distance map and contact map from pdb file."
    parser.add_argument("-f", "--fasta", help="input fasta file",type=is_file,required=True)
    parser.add_argument("-p", "--pdb", help="input pdb file",type=is_file,required=True)
    parser.add_argument("-o", "--outdir", help="output folder",type=str,required=True)
    parser.add_argument("-t", "--type", help="type of mulclass distance map[G, C, D, T]",type=str, default='G', required=False)

    args = parser.parse_args()
    fastafile = args.fasta
    pdbfile = args.pdb
    outdir = args.outdir 
    mulcalss_type = args.type 

    GLOABL_Path = sys.path[0]
    print(GLOABL_Path)


    pdb2dist = GLOABL_Path + '/pdb2dist.pl'

    tempdir = outdir + '/tempdist/'
    realdistdir = outdir + '/dist/'
    binarydir = outdir + '/bin/'
    if mulcalss_type == 'G':
        mulclassdir = outdir + '/mulclass_2_22/'
    elif mulcalss_type == 'C':
        mulclassdir = outdir + '/mulclass_4_20/'
    elif mulcalss_type == 'D':
        mulclassdir = outdir + '/mulclass_4_20/'
    elif mulcalss_type == 'T':
        mulclassdir = outdir + '/mulclass_3.5_19/'

    chkdirs(outdir)
    chkdirs(tempdir)
    chkdirs(realdistdir)
    chkdirs(binarydir)
    chkdirs(mulclassdir)

    target = os.path.basename(fastafile).split('.')[0]
    CAdist = tempdir + target + "_CA.dist"
    CBdist = tempdir + target + "_CB.dist"
    os.system("perl %s %s CA 0 8 > %s"%(pdb2dist, pdbfile, CAdist))
    os.system("perl %s %s CB 0 8 > %s"%(pdb2dist, pdbfile, CBdist))

    fasta = open(fastafile, 'r').readlines()[1].strip('\n')
    L = len(fasta)

    CB = dict()
    CA = dict()
    for line in open(tempdir+target+"_CB.dist","r"):
        line = line.rstrip()
        arr = line.split()
        CB[arr[0]+' '+arr[1]] = arr[2]
    for line in open(tempdir+target+"_CA.dist","r"):
        line = line.rstrip()
        arr = line.split()
        CA[arr[0]+' '+arr[1]] = arr[2]

    realdistfile = realdistdir + target + '.txt'
    binaryfile = binarydir + target + '.txt'
    mulclassfile = mulclassdir + target + '.txt'
    realdistmat = np.zeros((L,L))
    for i in range(0,L):
        for j in range(i+1,L):
            if (str(i+1)+" "+str(j+1)) in CB:
                realdistmat[i, j] = float(CB[str(i+1)+" "+str(j+1)])
            elif (str(i+1)+" "+str(j+1)) in CA:
                realdistmat[i, j] = float(CA[str(i+1)+" "+str(j+1)])
    realdistmat += realdistmat.T
    np.savetxt(realdistfile, realdistmat, fmt = '%.3f')
    binarymap = np.copy(realdistmat)
    binarymap[binarymap == 0] = 20
    binarymap[binarymap < 8] = 1
    binarymap[binarymap >= 8] = 0
    np.savetxt(binaryfile, binarymap, fmt = '%d')
    mulclassmat = real_value2mul_class(realdistmat, option = mulcalss_type)
    np.savetxt(mulclassfile, mulclassmat, fmt = '%d')
    # break

