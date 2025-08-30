#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores 
#SBATCH --gres=gpu:a100:1  # number of gpus
#SBATCH -t 6:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH --mem=16G
#SBATCH -o sysout/bench%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e sysout/bench%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="phjiang@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment
 

#Load required software
module load mamba/latest
#Activate our enviornment
source activate convSelfAtten
#Change to the directory of our script
cd /scratch/phjiang/csa_testing
#Run the software/python script
~/.conda/envs/convSelfAtten/bin/python bench.py

