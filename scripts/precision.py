# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:48:22 2017

@author: lee
"""
import numpy as np
import scipy.linalg as la 
import numpy.linalg as na
import os
import aaweights
import sys
def ROPE(S, rho):
    p=S.shape[0]
    S=S
    try:
        LM=na.eigh(S)
    except:
        LM=la.eigh(S)
    L=LM[0]
    M=LM[1]
    for i in range(len(L)):
        if L[i]<0:
            L[i]=0
    lamda=2.0/(L+np.sqrt(np.power(L,2)+8*rho))
    indexlamda=np.argsort(-lamda)
    lamda=np.diag(-np.sort(-lamda)[:p])
    hattheta=np.dot(M[:,indexlamda],lamda)
    hattheta=np.dot(hattheta,M[:,indexlamda].transpose())
    return hattheta
def blockshaped(arr,dim=21):
    p=arr.shape[0]//dim
    re=np.zeros([dim*dim,p,p])
    for i in range(p):
        for j in range(p):
            re[:,i,j]=arr[i*dim:i*dim+dim,j*dim:j*dim+dim].flatten()
    return re
def computepre(msafile,weightfile):
    msa=aaweights.read_msa(msafile)
    weights=np.genfromtxt(weightfile).flatten()
    cov=(aaweights.cal_large_matrix1(msa,weights))
    rho2=np.exp((np.arange(80)-60)/5.0)[30]
    pre=ROPE(cov,rho2)
    #print(pre)
    return blockshaped(pre)
def computeapre(msafile,weightfile,savefile):
    print(msafile)
    #if not os.path.isfile(savefile+'.npy222'):
    pre=computepre(msafile,weightfile)
    #pre=pre.astype('float32')
    #np.save(savefile,pre)
    pre.astype('float32').tofile(savefile)
def compute_fasta(fasta,updir,savedir):
    #updir='/oasis/projects/nsf/mia174/liyangum/deepPRE/makealn_test/'
    #savedir='/oasis/scratch/comet/liyangum/temp_project/pre_compute/'
    lines=open(fasta).readlines()
    pdbids=[lines[2*i][1:].strip() for i in range(len(lines)//2)] 
    #pool = Pool(processes=3)
    for pdbid in pdbids:
        msafile=updir+pdbid+'/'+pdbid+'.aln'
        weightfile=updir+pdbid+'/'+pdbid+'.weight'
        savefile=savedir+pdbid+'.pre'
        computeapre(msafile,weightfile,savefile)
        #pool.apply_async(computeapre,[msafile,weightfile,savefile])
    #pool.close()
    #pool.join()     
if __name__ == "__main__":    
    updir='/oasis/projects/nsf/mia174/liyangum/deepPRE/makealn_test/'
    savedir= '/oasis/scratch/comet/liyangum/temp_project/pre_compute/pre/'   
    #if not os.path.isdir(savedir):
    #    os.makedirs(savedir)
    inputfasta=sys.argv[1]
    compute_fasta(inputfasta,updir,savedir)