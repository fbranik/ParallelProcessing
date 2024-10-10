#!/bin/bash

## Give the Job a descriptive name
#PBS -N runFWSr

## Output and error files
#PBS -o fwSrPar.out
#PBS -e runFWSr.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=00:50:00

## Start
## Run make in the src folder (modify properly)

## export OMP_DYNAMIC=TRUE
export OMP_NESTED=true
export GOMP_CPU_AFFINITY="0-63"
			
module load openmp
cd /home/parallel/parlab07/a2/FW 
for j in 1 2 4 8 16 32 64 
do
	for k in 1024 2048 4096
	do
		for i in 64 128 512 
		do
			export OMP_NUM_THREADS=$j
			./fw_sr $k $i	
		done
	done
done
