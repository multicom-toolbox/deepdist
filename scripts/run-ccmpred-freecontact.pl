#!/usr/bin/perl -w
# Badri Adhikari, 5-21-2017

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;

my $db_tool_dir = shift;
my $aln            = shift;
my $CCMPREDdir     = shift;
my $FREECONTACTdir = shift;

####################################################################################################
use constant{
	FREECONTACT=> '/tools/freecontact-1.0.21/bin/freecontact',
	CCMPRED   => '/tools/CCMpred_plm/bin/ccmpred',
	HOURLIMIT => 24, ## modified by Jie 10/09/2017 for test
	NPROC     => 8
};

confess 'Oops!! CCMPRED  not found!'.$db_tool_dir.CCMPRED if not -f $db_tool_dir.CCMPRED;
confess 'Oops!! FREECONTACT not found!'.$db_tool_dir.FREECONTACT if not -f $db_tool_dir.FREECONTACT;


####################################################################################################
if (not $aln or not -f $aln){
	print "Alignment file $aln does not exist!\n" if ($aln and not -f $aln);
	print "Usage: $0 <aln-file> <CCMPRED-output-directory> <FREECONTACT-output-directory>\n";
	exit(1);
}
$aln = abs_path($aln);
if (not $CCMPREDdir){
	print 'CCMPRED Output directory not defined!';
	print "Usage: $0 <aln-file> <CCMPRED-output-directory> <FREECONTACT-output-directory>\n";
	exit(1);
}
system_cmd("mkdir -p $CCMPREDdir");
$CCMPREDdir = abs_path($CCMPREDdir);

if (not $FREECONTACTdir){
	print 'FREECONTACT Output directory not defined!';
	print "Usage: $0 <aln-file> <CCMPRED-output-directory> <FREECONTACT-output-directory>\n";
	exit(1);
}
system_cmd("mkdir -p $FREECONTACTdir");
$FREECONTACTdir = abs_path($FREECONTACTdir);

####################################################################################################
my $id = basename($aln, ".aln");
$aln = abs_path($aln);

# check and quit, if there are any results already
my $existing = `find $CCMPREDdir -name "*.rr" | wc -l`;
$existing = 0 if not $existing;
confess 'Oops!! There are already some rr files in the CCMPRED ouput directory! Consider running in an empty directory!' if int($existing) > 0;

####################################################################################################
print "Started [$0]: ".(localtime)."\n";

chdir $CCMPREDdir or confess $!;
system_cmd("cp $aln ./");
open  JOB, ">$id-ccmpred.sh" or confess "ERROR! Could not open $id-ccmpred.sh $!";
print JOB "#!/bin/bash\n";
print JOB "touch ccmpred.running\n";
print JOB "echo \"running ccmpred ..\"\n";
print JOB $db_tool_dir.CCMPRED." -t ".NPROC." $id.aln $id.ccmpred $id.plm > $id.ccmpred.log\n";
print JOB "if [ -s \"$id.ccmpred\" ]; then\n";
print JOB "   mv ccmpred.running ccmpred.done\n";
print JOB "   echo \"ccmpred job done.\"\n";
print JOB "   exit\n";
print JOB "fi\n";
print JOB "echo \"ccmpred failed!\"\n";
print JOB "mv ccmpred.running ccmpred.failed\n";
close JOB;
system_cmd("chmod 755 $id-ccmpred.sh");
print "Starting job $id-ccmpred.sh ..\n";
system "./$id-ccmpred.sh > $id.ccmpred.log &";
sleep 1;

####################################################################################################
chdir $FREECONTACTdir or confess $!;
system_cmd("cp $aln ./");

open  JOB, ">$id-freecontact.sh" or confess "ERROR! Could not open $id-freecontact.sh $!";
print JOB "#!/bin/bash\n";

#############export library for freecontact
my $gcc_v = `gcc -dumpversion`;
chomp $gcc_v;
my @gcc_version = split(/\./,$gcc_v);
if($gcc_version[0] != 4)
{
	print "!!!! Warning: gcc 4.X.X is recommended for boost installation, currently is $gcc_v\n\n";
	sleep(2);
	
}
if($gcc_version[0] ==4 and $gcc_version[1]<6) #gcc 4.6
{
    print JOB "export LD_LIBRARY_PATH=$db_tool_dir/tools/boost_1_38_0/lib/:$db_tool_dir/tools/OpenBLAS:\$LD_LIBRARY_PATH\n\n";
}else{
    print JOB "export LD_LIBRARY_PATH=$db_tool_dir/tools/boost_1_55_0/lib/:$db_tool_dir/tools/OpenBLAS:\$LD_LIBRARY_PATH\n\n";
}

print JOB "touch freecontact.running\n";
print JOB "echo \"running freecontact ..\"\n";
print JOB "".$db_tool_dir.FREECONTACT." < $id.aln > $id.freecontact.rr\n";
print JOB "if [ -s \"$id.freecontact.rr\" ]; then\n";
print JOB "   mv freecontact.running freecontact.done\n";
print JOB "   echo \"freecontact job done.\"\n";
print JOB "   exit\n";
print JOB "fi\n";
print JOB "echo \"freecontact failed!\"\n";
print JOB "mv freecontact.running freecontact.failed\n";
close JOB;
system_cmd("chmod 755 $id-freecontact.sh");
print "Starting job $id-freecontact.sh ..\n";
system "./$id-freecontact.sh &";
sleep 1;

####################################################################################################
print("\nWait for max ".HOURLIMIT." hours until all jobs are done ..\n");
my $running = 1;
my $i = 0;
while(int($running) > 0){
	sleep (HOURLIMIT);
	my $CCMPRED_running = `find $CCMPREDdir/ -name "*.running" | wc -l`;
	chomp $CCMPRED_running;
	$running = int($CCMPRED_running);
	last if $running == 0;
	$i++;
	last if $i == 3600;
}

####################################################################################################
print "\nChecking freecontact prediction..\n";
if(not -f "$FREECONTACTdir/$id.freecontact.rr"){
	#confess "Looks like CCMPRED did not finish! $FREECONTACTdir/$id.freecontact.rr is absent!\n";
	print "Looks like CCMPRED did not finish! $FREECONTACTdir/$id.freecontact.rr is absent!\n"; # modified by Jie 10/09/2017
	system_cmd("touch $FREECONTACTdir/$id.freecontact.rr");
}

####################################################################################################
print "\nChecking CCMPRED prediction..\n";
if(not -f "$CCMPREDdir/$id.ccmpred"){
	#confess "Looks like CCMPRED did not finish! $CCMPREDdir/$id.ccmpred is absent!\n";
	print "Looks like CCMPRED did not finish! $CCMPREDdir/$id.ccmpred is absent!\n"; # modified by Jie 10/09/2017
	system_cmd("touch $CCMPREDdir/$id.ccmpred");
}

print "\nFinished [$0]: ".(localtime)."\n";

####################################################################################################
sub system_cmd{
	my $command = shift;
	my $log = shift;
	confess "EXECUTE [$command]?\n" if (length($command) < 5  and $command =~ m/^rm/);
	if(defined $log){
		system("$command &> $log");
	}
	else{
		system($command);
	}
	if($? != 0){
		my $exit_code  = $? >> 8;
		confess "ERROR!! Could not execute [$command]! \nError message: [$!]";
	}
}
