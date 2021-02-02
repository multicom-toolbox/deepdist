#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Aug 02 

@author: tianqi
"""

docstring='''
DNCON4-setup for database and tools

usage: python setup.py
'''

import sys
import os
import re
import glob
import subprocess
from time import sleep

# Database version
uniref90_info = "uniref90_01_2020"
metaclust50_info = "2018_06"
hhsuitedb_type = "2020_01"
hhsuitedb_info = "UniRef30_2020_01"
ebi_uniref100_info = "myg_uniref100_01_2020"
uniref90_dir = "" #"/exports/store2/casp14/databases/deepmsa_db"
metaclust50_dir = "" #"/exports/store2/casp14/databases/deepmsa_db"
hhsuitedb_dir = "" #"/exports/store2/casp14/databases/deepmsa_db"
ebi_uniref100_dir =""


def boost_config(install_dir,tools_dir):
    if os.path.exists(install_dir+"/P1_install_boost.sh"):
        os.system("rm "+install_dir+"/P1_install_boost.sh")

    ##### check gcc version
    retcode = subprocess.call("gcc -dumpversion",shell=True);
    if retcode:
        print("Failed to find gcc in system, please check gcc version")
        sys.exit(1)

    gcc_v = subprocess.check_output(["gcc", "-dumpversion"]).strip()
    gcc_v = gcc_v.decode()
    gcc_version = gcc_v.split(".")
    
    if(int(gcc_version[0]) != 4):
        print("!!!! Warning: gcc 4.X.X is recommended for boost installation, currently is "+gcc_v+"\n\n")
        sleep(2)

    if(int(gcc_version[0]) ==4 and int(gcc_version[1])<6): #gcc 4.6
        print("\nGCC "+gcc_v+" is used, install boost-1.38.00\n\n")
        ### install boost-1.38 
        f = open(install_dir+"/P1_install_boost.sh","w")
        f.write("#!/bin/bash -e\n\n")
        f.write("echo \" Start compile boost (will take ~20 min)\"\n\n")
        f.write("cd "+tools_dir+"\n\n")
        f.write("cd boost_1_38_0\n\n")
        f.write("./configure  --prefix="+tools_dir+"/boost_1_38_0\n\n")
        f.write("make\n\n")
        f.write("make install\n\n")
        f.write("echo \"installed\" > "+tools_dir+"/boost_1_38_0/install.done\n\n")
        f.close() 
        #### install freecontact using boost 1.38
        f = open(install_dir+"/P3_install_freecontact.sh","w")
        f.write("#!/bin/bash -e\n\n")
        f.write("echo \" Start compile freecontact (will take ~1 min)\"\n\n")
        f.write("cd "+tools_dir+"\n\n")
        f.write("cd freecontact-1.0.21\n\n")
        f.write("autoreconf -f -i\n\n")
        f.write("./configure --prefix="+tools_dir+"/freecontact-1.0.21 LDFLAGS=\"-L"+tools_dir+"/OpenBLAS/lib -L"+tools_dir+"/boost_1_38_0/lib\" CFLAGS=\"-I"+tools_dir+"/OpenBLAS/include -I"+tools_dir+"/boost_1_38_0/include/boost-1_38\"  CPPFLAGS=\"-I"+tools_dir+"/OpenBLAS/include -I"+tools_dir+"s/boost_1_38_0/include/boost-1_38\" --with-boost="+tools_dir+"/boost_1_38_0/\n\n")
        f.write("make\n\n")
        f.write("make install\n\n")
        f.write("if [[ -f \"bin/freecontact\" ]]; then\n")
        f.write("\techo \"bin/freecontact exists\"\n")
        f.write("\techo \"installed\" > "+tools_dir+"/freecontact-1.0.21/install.done\n\n")
        f.write("else\n\n")
        f.write("\techo \"bin/freecontact doesn't exist, check the installation\"\n")
        f.write("fi\n\n")
        f.close()
    else:
        print("GCC "+gcc_v+" is used, install boost-1.55.00\n\n")
        ### install boost-1.55 
        f = open(install_dir+"/P1_install_boost.sh","w")
        f.write("#!/bin/bash -e\n\n")
        f.write("echo \" Start compile boost (will take ~20 min)\"\n\n")
        f.write("cd "+tools_dir+"\n\n")
        f.write("cd boost_1_55_0\n\n")
        f.write("./bootstrap.sh  --prefix="+tools_dir+"/boost_1_55_0\n\n")
        f.write("./b2\n\n")
        f.write("./b2 install\n\n")
        f.write("echo \"installed\" > "+tools_dir+"/boost_1_55_0/install.done\n\n")
        f.close()
        #### install freecontact using boost 1.55
        f = open(install_dir+"/P3_install_freecontact.sh","w")
        f.write("#!/bin/bash -e\n\n")
        f.write("echo \" Start compile freecontact (will take ~1 min)\"\n\n")
        f.write("cd "+tools_dir+"\n\n")
        f.write("cd freecontact-1.0.21\n\n")
        f.write("autoreconf -f -i\n\n")
        f.write("./configure --prefix="+tools_dir+"/freecontact-1.0.21 LDFLAGS=\"-L"+tools_dir+"/OpenBLAS/lib -L"+tools_dir+"/boost_1_55_0/lib\" CFLAGS=\"-I"+tools_dir+"/OpenBLAS/include -I"+tools_dir+"/boost_1_55_0/include\"  CPPFLAGS=\"-I"+tools_dir+"/OpenBLAS/include -I"+tools_dir+"/boost_1_55_0/include\" --with-boost="+tools_dir+"/boost_1_55_0/\n\n")
        f.write("make\n\n")
        f.write("make install\n\n")
        f.write("if [[ -f \"bin/freecontact\" ]]; then\n")
        f.write("\techo \"bin/freecontact exists\"\n")
        f.write("\techo \"installed\" > "+tools_dir+"/freecontact-1.0.21/install.done\n\n")
        f.write("else\n\n")
        f.write("\techo \"bin/freecontact doesn't exist, check the installation\"\n")
        f.write("fi\n\n")
        f.close()
    return gcc_version

def boost_install(install_dir,tools_dir,gcc_version):
    ### install boost-1.55 
    os.chdir(install_dir)
    if int(gcc_version[0]) ==4 and int(gcc_version[1])<6: #gcc 4.6
        if not os.path.exists(tools_dir+"/boost_1_38_0/install.done"):
            print("\nStart install boost_1.38, may take ~20 min (sh P1_install_boost.sh &> P1_install_boost.log)\n\n")
            print("\n\t\t\tLog is saved in "+install_dir+"/P1_install_boost.log\n\n")
            os.system("sh P1_install_boost.sh &> P1_install_boost.log")
            if os.path.exists(tools_dir+"/boost_1_55_0"):
                os.system("mv "+tools_dir+"/boost_1_55_0 "+tools_dir+"/boost_1_55_0_original")
                os.system("ln -s "+tools_dir+"/boost_1_38_0 "+tools_dir+"/boost_1_55_0")
        else:
            print("\nboost-1.38 is installed!\n\n")
    else:
        if not os.path.exists(tools_dir+"/boost_1_55_0/install.done"):
            print("\nStart install boost_1.55, may take ~20 min (sh P1_install_boost.sh &> P1_install_boost.log)\n\n")
            print("\n\t\t\tLog is saved in "+install_dir+"/P1_install_boost.log\n\n")
            os.system("sh P1_install_boost.sh &> P1_install_boost.log")
        else:
            print("\nboost-1.55 is installed!\n\n")

def OpenBlas_install(install_dir,tools_dir):
    #### install OpenBlas
    if  not os.path.exists(tools_dir+"/OpenBLAS/install.done"):
        print("\nStart install OpenBlas, may take ~1 min (sh P2_install_OpenBlas.sh &> P2_install_OpenBlas.log)\n\n")
        print("\n\t\t\tLog is saved in "+install_dir+"P2_install_OpenBlas.log\n\n")
        os.system("sh P2_install_OpenBlas.sh &> P2_install_OpenBlas.log")
    else:
        print("\nOpenBLAS is installed!\n\n")

def OpenBlas_config(install_dir,tools_dir):
    #### install OpenBlas cmd
    os.chdir(install_dir)
    f = open(install_dir+"/P2_install_OpenBlas.sh","w")
    f.write("#!/bin/bash -e\n\n")
    f.write("echo \" Start compile OpenBlas (will take ~5 min)\"\n\n")
    f.write("cd "+tools_dir+"\n\n")
    f.write("cd OpenBLAS\n\n")
    f.write("make\n\n")
    f.write("make PREFIX="+tools_dir+"/OpenBLAS install\n\n")
    f.write("echo \"installed\" > "+tools_dir+"/OpenBLAS/install.done\n\n")
    f.close()

def download_nr(db_tools_dir,tools_dir,identity):
    identity = str(identity)
    nr_dir = db_tools_dir+"/databases/nr"
    if not os.path.exists(nr_dir):
        os.makedirs(nr_dir)
    os.chdir(nr_dir)
    if os.path.exists("nr"+identity+".pal"):
        print("\tnr"+identity+" has been formatted, skip!")
    elif os.path.exists("nr"+identity):
        print("\tnr"+identity+" is found, start formating......")
        os.system(tools_dir+"/blast-2.2.26/bin/formatdb -i nr"+identity+" -o T -t nr"+identity+" -n nr"+identity)
        os.system("chmod -R 755 nr"+identity+"*")
    else:
        if os.path.exists("nr"+identity+"-2016.tar.gz"):
            os.system("rm nr"+identity+"-2016.tar.gz")
        os.system("wget http://sysbio.rnet.missouri.edu/dncon4_db_tools/databases/nr"+identity+"-2016.tar.gz")
        if os.path.exists("nr"+identity+"-2016.tar.gz"):
            print("\tnr"+identity+"-2016.tar.gz is found, start extracting files")
        else:
            print("Failed to download nr"+identity+" from http://sysbio.rnet.missouri.edu/dncon4_db_tools/databases/nr"+identity+"-2016.tar.gz")
            sys.exit(1)
        os.system("tar zxvf nr"+identity+"-2016.tar.gz && rm "+identity+"-2016.tar.gz")
        os.system(tools_dir+"/blast-2.2.26/bin/formatdb -i nr"+identity+" -o T -t nr"+identity+" -n nr"+identity)
        os.system("chmod -R 755 nr"+identity+"*")
        print("Downloading and formatting nr"+identity+"....Done")

def download_uniref90(db_tools_dir,version,db_dir=""):
    tools_dir = os.path.join(db_tools_dir,"tools")
    if db_dir == "":
        db_dir = os.path.join(db_tools_dir,"databases")
    version = str(version)
    uniref_dir = os.path.join(db_dir,version)
    if not os.path.exists(uniref_dir):
        os.makedirs(uniref_dir)
    os.chdir(uniref_dir)
    if os.path.exists("uniref90.pal") and os.path.exists("uniref90.ssi"):
        print("\tuniref90 is found, skip!")
    else:
        # if os.path.exists(version+".tar.gz"):
        #     os.system("rm "+version+".tar.gz")
        os.system("wget http://sysbio.rnet.missouri.edu/dncon4_db_tools/databases/"+version+".tar.gz")
        if os.path.exists(version+".tar.gz"):
            print("\tuniref90 is found, start extracting files")
        else:
            print("Failed to download uniref90 from http://sysbio.rnet.missouri.edu/dncon4_db_tools/databases/"+version+".tar.gz")
            sys.exit(1)
        os.system("tar zxvf "+version+".tar.gz && rm "+version+".tar.gz")
        #os.system("mv uniref"+identity+".fasta uniref"+identity)
        retcode = subprocess.call(tools_dir+"/hmmer-3.1b2-linux-intel-x86_64/easel/miniapps/esl-sfetch --index uniref90",shell=True)
        if retcode :
            print("Failed to index "+db_tools_dir+"/databases/uniref90")
            sys.exit(1)
        if os.path.exists("uniref90.pal"):
            print("\tuniref90 has been formatted, skip!")
        else:
            print("\tuniref90 starts formating......")
            os.system(tools_dir+"/blast-2.2.26/bin/formatdb -i uniref90 -o T -t uniref90 -n uniref90")
            os.system("chmod -R 755 uniref90*")
        print("Downloading and formatting uniref90....Done")

def download_metaclust50(db_tools_dir,version,db_dir=""):
    tools_dir = os.path.join(db_tools_dir,"tools")
    if db_dir == "":
        db_dir = os.path.join(db_tools_dir,"databases")
    version = str(version)
    metaclust_dir = os.path.join(db_dir,"Metaclust_"+version)
    if not os.path.exists(metaclust_dir):
        os.makedirs(metaclust_dir)
    os.chdir(metaclust_dir)
    if os.path.exists("metaclust_50") and os.path.exists("metaclust_50"+".ssi"):
        print("\tmetaclust_50"+" is found, skip!")
    else:
        if os.path.exists("metaclust_50.fasta.gz"):
            os.system("rm metaclust_50.fasta.gz")
        os.system("wget https://metaclust.mmseqs.org/"+version+"/metaclust_50.fasta.gz")
        if os.path.exists("metaclust_50.fasta.gz"):
            print("\tmetaclust_50 is found, start extracting files")
        else:
            print("Failed to download metaclust_50 from https://metaclust.mmseqs.org/"+version+"/metaclust_50.fasta.gz")
            sys.exit(1)
        os.system("gunzip metaclust_50.fasta.gz")
        os.system("mv metaclust_50.fasta metaclust_50")
        retcode = subprocess.call(tools_dir+"/hmmer-3.1b2-linux-intel-x86_64/easel/miniapps/esl-sfetch --index metaclust_50", shell=True)
        if retcode :
            print("Failed to index "+metaclust_dir)
            sys.exit(1)
        print("Downloading "+metaclust_dir+"....Done")

def download_ebi_uniref100(db_tools_dir,version,db_dir=""):
    tools_dir = os.path.join(db_tools_dir,"tools")
    if db_dir == "":
        db_dir = os.path.join(db_tools_dir,"databases")
    version = str(version)
    ebi_ref_dir = os.path.join(db_dir,version)
    if not os.path.exists(ebi_ref_dir):
        os.makedirs(ebi_ref_dir)
    os.chdir(ebi_ref_dir)
    if os.path.exists("myg_uniref100") and os.path.exists("myg_uniref100"+".ssi"):
        print("\tmyg_uniref100"+" is found, skip!")
    else:
        if os.path.exists(version+".tar.gz"):
            os.system("rm "+version+".tar.gz")
        os.system("wget http://sysbio.rnet.missouri.edu/dncon4_db_tools/databases/"+version+".tar.gz")
        if os.path.exists(version+".tar.gz"):
            print("\tmyg_uniref100 is found, start extracting files")
        else:
            print("Failed to download myg_uniref100 from http://sysbio.rnet.missouri.edu/dncon4_db_tools/databases/"+version+".tar.gz")
            sys.exit(1)
        os.system("tar zxvf "+version+".tar.gz && rm "+version+".tar.gz")
        retcode = subprocess.call(tools_dir+"/hmmer-3.1b2-linux-intel-x86_64/easel/miniapps/esl-sfetch --index myg_uniref100", shell=True)
        if retcode :
            print("Failed to index "+ebi_ref_dir)
            sys.exit(1)
        print("Downloading "+ebi_ref_dir+"....Done")

def direct_download(tool, address, tools_dir):  ####Tools don't need to be configured after downloading and configuring
    os.chdir(tools_dir)
    if not os.path.exists(tools_dir+"/"+tool):
        os.system("wget "+address)
    if os.path.exists(tools_dir+"/"+tool):
        print("Decompressing "+tools_dir+"/"+tool)
        if "tar.gz" in tool:
            os.system("tar -zxf "+tool+" && rm "+tool)
            tool = re.sub("\.tar.gz","",tool)
        if "zip" in tool:
            os.system("unzip "+tool)
            tool = re.sub("\.zip","",tool)
        if tool == "blast-2.2.26-x64-linux":
            tool = "blast-2.2.26"
        if tool == "package":
            tool = "hhsuite2"
        os.system("chmod -R 755 "+tool)
        print("Downloading "+tools_dir+"/"+tool+"....Done")
    else:
        print("Failed to download "+tool+" from "+address)
        sys.exit(1)

#configure_file(filepath, filetype, 'feature_dir', db_dir)
def configure_file(filepath, filetype, keyword, db_dir):
    os.chdir(filepath)
    for filename in glob.glob(filepath + '/*.' + filetype):
        temp_in = filename
        print(temp_in)
        temp_out = temp_in+'.tmp'
        f = open(temp_in, 'r')
        tar_flag = False
        line_old = None
        line_new = None
        for line in f.readlines():
            if 'FLAG' in line:
                tar_flag = True
            if keyword in line and tar_flag == True:
                tar_flag = False
                line_old = line.strip('\n')
                fix_str = line.strip('\n').split('=')[0]
                line_new = fix_str + '=' + db_dir
                # print(line_new)s
        f.close()
        #replace target line
        f1 = open(temp_in)
        con = f1.read()
        f1.close()
        con_new = con.replace(line_old, line_new)
        f2 = open(temp_out, 'w')
        f2.write(con_new)
        f2.close()
        os.system('mv ' + temp_out + ' ' + temp_in)


if __name__ == '__main__':
    argv=[]
    for arg in sys.argv[1:]:
        if arg.startswith("-h"):
            print(docstring)

    # Set directory of multicom databases and tools
    # db_tools_dir = "/storage/htc/bdm/zhiye/DNCON4_db_tools/" 
    temp_path = os.path.dirname(os.path.abspath(__file__))
    os.environ["TMPDIR"] = temp_path+"/tmp"
    if not os.path.exists(temp_path+"/tmp"):
        os.makedirs(temp_path+"/tmp")

    if sys.version_info[0] < 3:
        intall_flag = raw_input("Intall DeepDist_db_tools to "+ temp_path +" ? (Yes/No)")
        if 'Y' in intall_flag or 'y' in intall_flag:
            db_tools_dir = temp_path
        elif 'N' in intall_flag or 'n' in intall_flag:
            custom_path = raw_input("Please input the path of DeepDist_db_tools you want to install...\n")
            print("The DeepDist_db_tools will be installed to %s, please wait...\n"%custom_path)
            db_tools_dir = custom_path
        else:
            print("Input illeage! System exit!")
            sys.exit(1)
    else:
        intall_flag = input("Intall DeepDist_db_tools to "+ temp_path +" ? (Yes/No)")
        if 'Y' in intall_flag or 'y' in intall_flag:
            db_tools_dir = temp_path
        elif 'N' in intall_flag or 'n' in intall_flag:
            custom_path = input("Please input the path of DeepDist_db_tools you want to install...\n")
            print("The DeepDist_db_tools will be installed to %s, please wait...\n"%custom_path)
            db_tools_dir = custom_path
        else:
            print("Input illeage! System exit!")
            sys.exit(1)


    ##!!! Don't Change the code below##
    global_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(global_path):
        print("The DNCON4 directory "+global_path+" is not existing, set the path as your unzipped DNCON4 directory")
        sys.exit(1)

    if not os.path.exists(db_tools_dir):
        os.makedirs(db_tools_dir)

    db_tools_dir=os.path.abspath(db_tools_dir)

    # write db_dir to /installation/path.inf
    configure_file(global_path + '/installation/', 'inf', 'DeepDist_db_dir', db_tools_dir)

    os.chdir(global_path)
    script_path = os.path.dirname(os.path.abspath(__file__))
    # install virtual environment
    vir_dir = global_path+ '/env'
    if not os.path.exists(vir_dir):
        os.makedirs(vir_dir)
        os.system("chmod -R 755 "+vir_dir)
    if os.path.exists(vir_dir+"/env_vir.done"):
        print(vir_dir+"/env_vir installed....skip")
    else:
        os.system("touch "+vir_dir+"/env_vir.running")
        retcode = subprocess.call("sh "+script_path+"/installation/set_env.sh", shell=True)
        if retcode :
            print("Failed to set virtual environment.... ")
            sys.exit(1)
        os.system("mv "+vir_dir+"/env_vir.running "+vir_dir+"/env_vir.done")
        print(vir_dir+"/env_vir installed")


    print("Start install DNCON4_db_tools into "+db_tools_dir)
    os.chdir(db_tools_dir)
    database_dir = db_tools_dir+"/databases"
    tools_dir = db_tools_dir+"/tools"
    install_dir = db_tools_dir+"/installation"


    if not os.path.exists(database_dir):
        os.makedirs(database_dir)
        os.system("chmod -R 755 "+database_dir)

    if not os.path.exists(tools_dir):
        os.makedirs(tools_dir)
        os.system("chmod -R 755 "+tools_dir)

    if not os.path.exists(install_dir):
        os.makedirs(install_dir)
        os.system("chmod -R 755 "+install_dir)

    #### Download blast-2.2.26 for formating sequence DB
    if os.path.exists(install_dir+"/blast-2.2.26.done"):
        print(install_dir+"/blast-2.2.26 installed....skip")
    else:
        os.system("touch "+install_dir+"/blast-2.2.26.running")
        tool = "blast-2.2.26-x64-linux.tar.gz" 
        address = "http://sysbio.rnet.missouri.edu/dncon4_db_tools/tools/blast-2.2.26-x64-linux.tar.gz" 
        direct_download(tool, address, tools_dir)
        os.system("mv "+install_dir+"/blast-2.2.26.running "+install_dir+"/blast-2.2.26.done")
        print(install_dir+"/blast-2.2.26 installed")

    #### Download hmmer-3.1b2-linux-intel-x86_64 for indexing sequence DB
    if os.path.exists(install_dir+"/hmmer-3.1b2-linux-intel-x86_64.done"):
        print(install_dir+"/hmmer-3.1b2-linux-intel-x86_64 installed....skip")
    else:
        os.system("touch "+install_dir+"/hmmer-3.1b2-linux-intel-x86_64.running")
        tool = "hmmer-3.1b2-linux-intel-x86_64.tar.gz"
        address = "http://eddylab.org/software/hmmer3/3.1b2/hmmer-3.1b2-linux-intel-x86_64.tar.gz"
        direct_download(tool, address, tools_dir)
        tool = re.sub("\.tar.gz","",tool)
        os.chdir(tools_dir+"/"+tool)
        retcode = subprocess.call("./configure", shell=True)
        if retcode :
            print("Failed to configure "+tools_dir+"/"+tool)
            sys.exit(1)
        retcode = subprocess.call("make", shell=True)
        if retcode :
            print("Failed to make "+tools_dir+"/"+tool)
            sys.exit(1)
        os.system("mv "+install_dir+"/hmmer-3.1b2-linux-intel-x86_64.running "+install_dir+"/hmmer-3.1b2-linux-intel-x86_64.done")
        print(install_dir+"/hmmer-3.1b2-linux-intel-x86_64 installed")

    ### (2) Download databases
    os.chdir(database_dir)

    #### Download Uniref90
    print("Download Uniref90\n");
    download_uniref90(db_tools_dir,uniref90_info,uniref90_dir)

    ### (3) Download basic tools
    #### Downlaod SCRATCH
    if os.path.exists(install_dir+"/SCRATCH-1D_1.1.done"):
        print(install_dir+"/SCRATCH-1D_1.1 installed....skip")
    else:
        os.system("touch "+install_dir+"/SCRATCH-1D_1.1.running")
        tool = "SCRATCH-1D_1.1.tar.gz"
        address = "wget http://download.igb.uci.edu/SCRATCH-1D_1.1.tar.gz"
        direct_download(tool, address, tools_dir)
        tool = re.sub("\.tar.gz","",tool)
        os.chdir(tools_dir+"/"+tool)
        print(tools_dir+"/"+tool)
        retcode = subprocess.call("perl install.pl", shell=True)
        if retcode :
            print("Failed to install "+tools_dir+"/"+tool)
            sys.exit(1)
        retcode = subprocess.call("mv ./pkg/blast-2.2.26 ./pkg/blast-2.2.26.original", shell=True)
        retcode = subprocess.call("cp -r "+tools_dir+"/blast-2.2.26 ./pkg/", shell=True)
        os.system("mv "+install_dir+"/SCRATCH-1D_1.1.running "+install_dir+"/SCRATCH-1D_1.1.done")
        print(install_dir+"/SCRATCH-1D_1.1 installed")

    #### Downlaod MetaPSICOV
    if os.path.exists(install_dir+"/metapsicov.done"):
        print(install_dir+"/metapsicov installed....skip")
    else:
        os.system("touch "+install_dir+"/metapsicov.running")
        tool = "metapsicov.tar.gz"
        address = "http://sysbio.rnet.missouri.edu/dncon4_db_tools/tools/metapsicov.tar.gz"
        direct_download(tool, address, tools_dir)
        os.system("mv "+install_dir+"/metapsicov.running "+install_dir+"/metapsicov.done")
        print(install_dir+"/metapsicov installed")

    #### Downlaod PSIPRED
    if os.path.exists(install_dir+"/psipred.4.0.done"):
        print(install_dir+"/psipred.4.0 installed....skip")
    else:
        os.system("touch "+install_dir+"/psipred.4.0.running")
        tool = "psipred.4.0.tar.gz"
        address = "http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/old_versions/psipred.4.0.tar.gz"
        direct_download(tool, address, tools_dir)
        os.system("mv "+install_dir+"/psipred.4.0.running "+install_dir+"/psipred.4.0.done")
        print(install_dir+"/psipred.4.0 installed")

    #### Download CCMpred_plm
    if os.path.exists(install_dir+"/CCMpred_plm.done"):
        print(install_dir+"/CCMpred_plm installed....skip")
    else:
        os.system("touch "+install_dir+"/CCMpred_plm.running")
        tool = "CCMpred_plm.tar.gz"
        address = "http://sysbio.rnet.missouri.edu/dncon4_db_tools/tools/CCMpred_plm.tar.gz"
        direct_download(tool, address, tools_dir)
        os.system("mv "+install_dir+"/CCMpred_plm.running "+install_dir+"/CCMpred_plm.done")
        print(install_dir+"/CCMpred_plm installed")

    #### Download CD-HIT
    if os.path.exists(install_dir+"/cd-hit-v4.6.8-2017-1208.done"):
        print(install_dir+"/cd-hit-v4.6.8-2017-1208 installed....skip")
    else:
        os.system("touch "+install_dir+"/cd-hit-v4.6.8-2017-1208.running")
        tool = "cd-hit-v4.6.8-2017-1208-source.tar.gz"
        address = "https://github.com/weizhongli/cdhit/releases/download/V4.6.8/cd-hit-v4.6.8-2017-1208-source.tar.gz"
        direct_download(tool, address, tools_dir)
        os.system("mv "+install_dir+"/cd-hit-v4.6.8-2017-1208.running "+install_dir+"/cd-hit-v4.6.8-2017-1208.done")
        print(install_dir+"/cd-hit-v4.6.8-2017-1208 installed")

    #### Download DeepAlign1.0
    if os.path.exists(install_dir+"/DeepAlign1.0.done"):
        print(install_dir+"/DeepAlign1.0 installed....skip")
    else:
        os.system("touch "+install_dir+"/DeepAlign1.0.running")
        tool = "DeepAlign1.0.tar.gz"
        address = "http://sysbio.rnet.missouri.edu/dncon4_db_tools/tools/DeepAlign1.0.tar.gz"
        direct_download(tool, address, tools_dir)
        os.system("mv "+install_dir+"/DeepAlign1.0.running "+install_dir+"/DeepAlign1.0.done")
        print(install_dir+"/DeepAlign1.0 installed")


