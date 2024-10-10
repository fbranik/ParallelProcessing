#!/bin/bash

## Give the Job a descriptive name
#PBS -N runFW

## Output and error files
#PBS -o fwStandard.out
#PBS -e runFWStandard.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=01:00:00

## Start
## Run make in the src folder (modify properly)

## export OMP_DYNAMIC=TRUE
module load openmp
cd /home/parallel/parlab07/a2/FW 
for j in 1024 2048 4096
do
##	export GOMP_CPU_AFFINITY="0-$(($j-1))"
##	export OMP_NUM_THREADS=$j
	./fw $j	
done
