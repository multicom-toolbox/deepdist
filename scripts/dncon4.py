#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:16:36 2019

@author: tianqi
"""

import os
import re
import sys

if __name__ == '__main__':
    if len(sys.argv)!=4:
        print("dncon4.py [db_tool_dir] [fasta_file] [outdir]")
        exit(1)

    db_tool_dir = os.path.abspath(sys.argv[1])
    fasta = os.path.abspath(sys.argv[2])
    outdir = os.path.abspath(sys.argv[3])
    
    script_path = os.path.dirname(os.path.abspath(__file__))
    target = os.path.basename(fasta)
    target = re.sub("\.fasta","",target)

    if not os.path.exists(fasta):
        print("Cannot fasta file:"+fasta)
        sys.exit(1)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print("Create output folder path:"+outdir)

    #step1: generate alignment
    if os.path.exists(outdir+"/alignment/"+target+".aln") and os.path.getsize(outdir+"/alignment/"+target+".aln") > 0:
        print("alignment generated.....skip")
    else:
        os.system(db_tool_dir+"/tools/DeepAlign1.0/hhjack_hhmsearch3.sh "+fasta+" "+outdir+"/alignment "+db_tool_dir+"/tools/ "+db_tool_dir+"/databases/")
        if os.path.exists(outdir+"/alignment/"+target+".aln") and os.path.getsize(outdir+"/alignment/"+target+".aln") > 0:
            print("alignment generated successfully....")
        else:
            print("alignment generation failed....")

    #step2: generate other features
    if os.path.exists(outdir+"/X-"+target+".txt") and os.path.getsize(outdir+"/X-"+target+".txt") > 0:
        print("DNCON2 features generated.....skip")
    else:
        os.system("perl "+script_path+"/generate-other.pl "+db_tool_dir+" "+fasta+" "+outdir)
        if os.path.exists(outdir+"/X-"+target+".txt") and os.path.getsize(outdir+"/X-"+target+".txt") > 0:
            print("DNCON2 features generated successfully....")
        else:
            print("DNCON2 features generation failed....")

    #step3: generate cov
    if os.path.exists(outdir+"/"+target+".cov") and os.path.getsize(outdir+"/"+target+".cov") > 0:
        print("cov generated.....skip")
    else:
        os.system(script_path+"/cov21stats "+outdir+"/alignment/"+target+".aln "+outdir+"/"+target+".cov")
        if os.path.exists(outdir+"/"+target+".cov") and os.path.getsize(outdir+"/"+target+".cov") > 0:
            print("cov generated successfully....")
        else:
            print("cov generation failed....")

    #step4: generate plm
    if os.path.exists(outdir+"/ccmpred/"+target+".plm") and os.path.getsize(outdir+"/ccmpred/"+target+".plm") > 0:
        print("plm generated.....skip")
        os.system("mv "+outdir+"/ccmpred/"+target+".plm "+outdir)
    elif os.path.exists(outdir+"/"+target+".plm") and os.path.getsize(outdir+"/"+target+".plm") > 0:
        print("plm generated.....skip")
    else:
        print("plm generation failed....")

    #step5: generate pre
    if os.path.exists(outdir+"/"+target+".pre") and os.path.getsize(outdir+"/"+target+".pre") > 0:
        print("pre generated.....skip")
    else:
        os.system(script_path+"/calNf_ly "+outdir+"/alignment/"+target+".aln 0.8 > "+outdir+"/"+target+".weight")
        os.system("python -W ignore "+script_path+"/generate_pre.py "+outdir+"/alignment/"+target+".aln "+outdir+"/"+target+" >"+outdir+"/pre.log")
        os.system("rm "+outdir+"/"+target+".weight")
        if os.path.exists(outdir+"/"+target+".pre") and os.path.getsize(outdir+"/"+target+".pre") > 0:
            print("pre generated successfully....")
        else:
            print("pre generation failed....")