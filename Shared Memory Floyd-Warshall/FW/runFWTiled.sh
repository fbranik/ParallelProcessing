#!/bin/bash

## Give the Job a descriptive name
#PBS -N runFWTiled

## Output and error files
#PBS -o fwTiled.out
#PBS -e runFWTiled.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=00:15:00

## Start
## Run make in the src folder (modify properly)

## export OMP_DYNAMIC=TRUE
module load openmp
cd /home/parallel/parlab07/a2/FW 
for j in 1024 2048 4096
do
	for i in 32 64 128 256 512
	do
	##	export GOMP_CPU_AFFINITY="0-$(($j-1))"
	##	export OMP_NUM_THREADS=$j
		./fw_tiled $j $i	
	done
done
