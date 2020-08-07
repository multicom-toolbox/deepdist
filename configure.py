import sys,os,glob,re

#configure_file(filepath, filetype, 'feature_dir', db_dir)
def configure_file(filepath, filetype, flag, keyword, db_dir):
    os.chdir(filepath)
    for filename in glob.glob(filepath + '/*.' + filetype):
        temp_in = filename
        temp_out = temp_in+'.tmp'
        f = open(temp_in, 'r')
        tar_flag = False
        change_flag = False
        line_old = None
        line_new = None
        for line in f.readlines():
            if flag in line:
                tar_flag = True
            if keyword in line and tar_flag == True:
                tar_flag = False
                change_flag = True
                line_old = line.strip('\n')
                fix_str = line.strip('\n').split('=')[0]
                if '\'' in line:
                    fix_str2 = line.strip('\n').split('\'')[-1]
                    line_new = fix_str + '=' + db_dir + '\'' + fix_str2
                else:
                    line_new = fix_str + '=' + db_dir
                # print(line_old)
                # print(line_new)
        f.close()
        #replace target line
        if change_flag:
            print(temp_in)
            change_flag = False
            f1 = open(temp_in)
            con = f1.read()
            f1.close()
            con_new = con.replace(line_old, line_new)
            f2 = open(temp_out, 'w')
            f2.write(con_new)
            f2.close()
            os.system('mv ' + temp_out + ' ' + temp_in)
            os.system('chmod -R 777 ' + temp_in)

def configure_database(filepath, filetype, flag, keyword, db_dir):
    os.chdir(filepath)
    for filename in glob.glob(filepath + '/*.' + filetype):
        temp_in = filename
        temp_out = temp_in+'.tmp'
        f = open(temp_in, 'r')
        tar_flag = False
        change_flag = False
        line_old = []
        line_new = []
        for line in f.readlines():
            if flag in line:
                tar_flag = True
            if '#####' in line:
                tar_flag = False
            for i in range(len(keyword)):
                if keyword[i] in line and tar_flag == True:
                    change_flag = True
                    line_old.append(line.strip('\n'))
                    fix_str = line.strip('\n').split('=')[0]
                    line_new.append(fix_str + '=' + '\'' + db_dir[i] + '\'')
        f.close()
        #replace target line
        if change_flag:
            print(temp_in)
            change_flag = False
            f1 = open(temp_in)
            con = f1.read()
            f1.close()
            for i in range(len(line_old)):
                if i == 0:
                    con_new = con.replace(line_old[i], line_new[i])
                else:
                    con_new = con_new.replace(line_old[i], line_new[i])
            f2 = open(temp_out, 'w')
            f2.write(con_new)
            f2.close()
            os.system('mv ' + temp_out + ' ' + temp_in)
            os.system('chmod -R 777 ' + temp_in)


temp_path = sys.path[0]
DeepDist_path = ''
if sys.version_info[0] < 3:
    intall_flag = raw_input("Intall DeepDist to "+ temp_path +". Press any button to continue...")
else:
    intall_flag = input("Intall DeepDist to "+ temp_path +". Press any button to continue...")
DeepDist_path = temp_path
    # if 'Y' in intall_flag or 'y' in intall_flag:
    #     DeepDist_path = temp_path
    # else:
    #     custom_path = input("Please input the path you want to install...")
    #     print("The DeepDist will be installed to %s, please wait...\n"%custom_path)
    #     DeepDist_path = custom_path
	## copy all file to the custom path, then need to change all shell gloable_dir

install_info_file = DeepDist_path+'/installation/path.inf'
DeepDist_db_dir =''
uniref90_dir=''
metaclust50_dir=''
hhsuitedb_dir=''
ebi_uniref100_dir=''
DeepDist_run_file=''
DeepDist_train_script=''
DeepDist_evalu_ensemble=''
DeepDist_evalu_individual=''
DeepDist_pred_ensemble=''
DeepDist_pred_individual=''
DeepDist_feature_generate= ''
if not os.path.exists(install_info_file):
    print("Can't find %s, please check!"%install_info_file)
    sys.exit(1)
else:
    f = open(install_info_file, 'r')
    for line in f.readlines():
        if line.startswith('#'):
            continue
        else:
            if 'db_dir' in line:
                DeepDist_db_dir = line.strip('\n').split('=')[1]
                if ' ' in DeepDist_db_dir:DeepDist_db_dir.replace(' ','')
                if 'none' in DeepDist_db_dir:
                	print("Database path hasn't set, please run setup.py!")
                	sys.exit(1)
            elif 'uniref90_version' in line:
                uniref90_version = line.strip('\n').split('=')[1]
                if ' ' in uniref90_version:uniref90_version.replace(' ','')
                uniref90_dir = DeepDist_db_dir + '/databases/' + uniref90_version
            elif 'metaclust50_version' in line:
                metaclust50_version = line.strip('\n').split('=')[1]
                if ' ' in metaclust50_version:metaclust50_version.replace(' ','')
                metaclust50_dir = DeepDist_db_dir + '/databases/' + metaclust50_version
            elif 'hhsuitedb_version' in line:
                hhsuitedb_version = line.strip('\n').split('=')[1]
                if ' ' in hhsuitedb_version:hhsuitedb_version.replace(' ','')
                hhsuitedb_dir = DeepDist_db_dir + '/databases/' + hhsuitedb_version
            elif 'ebi_uniref100_version' in line:
                ebi_uniref100_version = line.strip('\n').split('=')[1]
                if ' ' in ebi_uniref100_version:ebi_uniref100_version.replace(' ','')
                ebi_uniref100_dir = DeepDist_db_dir + '/databases/' + ebi_uniref100_version
            elif 'run_file' in line:
                DeepDist_run_file = line.strip('\n').split('=')[1]
                if ' ' in DeepDist_run_file:DeepDist_run_file.replace(' ','')
                DeepDist_run_file = DeepDist_path +'/' + DeepDist_run_file
            elif 'evalu_ensemble' in line:
                DeepDist_evalu_ensemble = line.strip('\n').split('=')[1]
                if ' ' in DeepDist_evalu_ensemble:DeepDist_evalu_ensemble.replace(' ','')
                DeepDist_evalu_ensemble = DeepDist_path +'/' + DeepDist_evalu_ensemble
            elif 'evalu_individual' in line:
                DeepDist_evalu_individual = line.strip('\n').split('=')[1]
                if ' ' in DeepDist_evalu_individual:DeepDist_evalu_individual.replace(' ','')
                DeepDist_evalu_individual = DeepDist_path +'/' + DeepDist_evalu_individual
            elif 'pred_ensemble' in line:
                DeepDist_pred_ensemble = line.strip('\n').split('=')[1]
                if ' ' in DeepDist_pred_ensemble:DeepDist_pred_ensemble.replace(' ','')
                DeepDist_pred_ensemble = DeepDist_path +'/' + DeepDist_pred_ensemble
            elif 'pred_individual' in line:
                DeepDist_pred_individual = line.strip('\n').split('=')[1]
                if ' ' in DeepDist_pred_individual:DeepDist_pred_individual.replace(' ','')
                DeepDist_pred_individual = DeepDist_path +'/' + DeepDist_pred_individual
            # elif 'feature_generate' in line:
            #     DeepDist_feature_generate = line.strip('\n').split('=')[1]
            #     if ' ' in DeepDist_feature_generate:DeepDist_feature_generate.replace(' ','')
            #     DeepDist_feature_generate = DeepDist_path +'/' + DeepDist_feature_generate


print("### Find database folder %s"%DeepDist_db_dir)
print("configure run file...")
configure_file(DeepDist_run_file, 'py', 'GLOBAL_FLAG', 'global_dir', DeepDist_path)
configure_file(DeepDist_run_file, 'py', 'DBTOOL_FLAG', 'db_tool_dir', DeepDist_db_dir)
configure_database(DeepDist_run_file, 'py', 'DATABASE_FLAG', ['uniref90_dir','metaclust50_dir','hhsuitedb_dir','ebi_uniref100_dir'], 
    [uniref90_dir,metaclust50_dir,hhsuitedb_dir,ebi_uniref100_dir])
print("configure evaluate ensemble script...")
configure_file(DeepDist_evalu_ensemble, 'sh', 'GLOBAL_FLAG', 'global_dir', DeepDist_path)
configure_file(DeepDist_evalu_ensemble, 'sh', 'FEATURE_FLAG', 'feature_dir', DeepDist_db_dir)
print("configure evaluate individual script...")
configure_file(DeepDist_evalu_individual, 'sh', 'GLOBAL_FLAG', 'global_dir', DeepDist_path)
configure_file(DeepDist_evalu_individual, 'sh', 'FEATURE_FLAG', 'feature_dir', DeepDist_db_dir)
print("configure predict ensemble script...")
configure_file(DeepDist_pred_ensemble, 'sh', 'GLOBAL_FLAG', 'global_dir', DeepDist_path)
configure_file(DeepDist_pred_ensemble, 'sh', 'DBTOOL_FLAG', 'db_tool_dir', DeepDist_db_dir)
print("configure predict individual script...")
configure_file(DeepDist_pred_individual, 'sh', 'GLOBAL_FLAG', 'global_dir', DeepDist_path)
configure_file(DeepDist_pred_individual, 'sh', 'DBTOOL_FLAG', 'db_tool_dir', DeepDist_db_dir)
# print("configure feature generate script...")
# configure_file(DeepDist_feature_generate, 'sh', 'GLOBAL_FLAG', 'global_dir', DeepDist_path)
# configure_file(DeepDist_feature_generate, 'sh', 'FEATURE_FLAG', 'feature_dir', DeepDist_db_dir)

### ask if want to donwload the feature data
if os.path.exists(DeepDist_db_dir):
    if os.path.exists(DeepDist_db_dir+'/features/'):
        os.chdir(DeepDist_db_dir+'/features/')
    else:
        os.mkdir(DeepDist_db_dir+'/features/')
    if sys.version_info[0] < 3:
        download_flag = raw_input("\n\nWould you want to download the training database ? It will take 30 min to download, and cost 400GB disk space. (No/Yes)")
    else:
        download_flag = input("Would you want to download the training database ? It will take 30 min to download, and cost 400GB disk space. (No/Yes)")
    if 'Y' in download_flag or 'y' in download_flag:
        os.system("wget http://sysbio.rnet.missouri.edu/dncon4_db_tools/features/DEEPCOV.tar.gz")
        if os.path.exists("DEEPCOV.tar.gz"):
            print("DEEPCOV.tar.gz already exists")
        else:
            print("Failed to download DEEPCOV.tar.gz from http://sysbio.rnet.missouri.edu/dncon4_db_tools/features/DEEPCOV.tar.gz")
            sys.exit(1)
    if sys.version_info[0] < 3:
        extract_flag = raw_input("Would you want to extract the training database ? It will take 15 hr to extract, and cost another 700GB disk space. (No/Yes)")
    else:
        extract_flag = input("Would you want to extract the training database ? It will take 15 hr to extract, and cost another 700GB disk space. (No/Yes)")
    if 'Y' in extract_flag or 'y' in extract_flag:
        os.system("tar zxvf DEEPCOV.tar.gz")
        print("Extract DEEPCOV.tar.gz done!")
        if os.path.exists("DEEPCOV.tar.gz"):
            os.system("rm DEEPCOV.tar.gz")
