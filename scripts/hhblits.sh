#!/bin/bash
export HHLIB=/storage/htc/bdm/zhiye/DNCON4_db_tools/tools/hhsuite-3.2.0-SSE2-Linux
PATH=$PATH:$HHLIB/bin:$HHLIB/scripts

if [ $# -ne 5 ]
then
        echo "need three parameters: target id, fasta file, output directory, db, coverage";
        exit 1
fi

[ -d $3 ] || mkdir $3
cd $3

if [ -f "${1}.aln" ]; then
    echo "hhblits job hhb-cov${4} generated ... Skip!"
    exit
fi


touch hhb-cov$5.running
echo "running hhblits job hhb-cov${5}.."

hhblits -i $2 -d $4 -oa3m $1.a3m -cpu 8 -n 3 -diff inf -e 0.001 -id 99 -cov $5 > hhb-cov$5-hhblits.log
cp $1.a3m hhb-cov$5.a3m
if [ ! -f "${1}.a3m" ]; then
   mv hhb-cov$5.running hhb-cov$5.failed
   echo "hhblits job hhb-cov${5} failed!"
   exit
fi
egrep -v "^>" $1.a3m | sed 's/[a-z]//g' > $1.aln
if [ -f "${1}.aln" ]; then
   mv hhb-cov$5.running hhb-cov$5.done
   echo "hhblits hhb-cov${5} job done."
   exit
fi
echo "Something went wrong! hhb-cov${5}.aln file not present!"
mv hhb-cov$5.running hhb-cov$5.failed

#sh /storage/htc/bdm/tianqi/CASP14/hhblits.sh T1024  /storage/htc/bdm/tianqi/CASP14/fasta/T1024.fasta T1024 /storage/htc/bdm/zhiye/DNCON4_db_tools/databases/UniRef30_2020_03/UniRef30_2020_03 50