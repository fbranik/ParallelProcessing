#!/bin/bash

## Give the Job a descriptive name
#PBS -N runNaive2

## Output and error files
#PBS -o runSeq.out
#PBS -e runSeq.err

## How many machines should we get?
#PBS -l nodes=1:ppn=1

##How long should the job run for?
#PBS -l walltime=01:00:00

## Start
## Run make in the src folder (modify properly)

## export OMP_DYNAMIC=TRUE
## export GOMP_CPU_AFFINITY="0-63"
## module load openmp
cd /home/parallel/parlab07/a2/kmeans 
./kmeans_seq -s 256 -n 16 -c 16 -l 10	
