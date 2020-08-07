#!/bin/bash -e

srun -p Lewis,hpc4,hpc5 --mem 20G -t 10:00:00 --pty bash
module load python/python-3.5.2

