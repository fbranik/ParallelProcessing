#!/bin/bash

## Give the Job a descriptive name
#PBS -N runNaive

## Output and error files
#PBS -o kmeansRevisedBonusTesting.out
#PBS -e kmeansRevised.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=00:05:00

## Start
## Run make in the src folder (modify properly)

## export OMP_DYNAMIC=TRUE
module load openmp
cd /home/parallel/parlab07/a21Revised
for i in kmeans_omp_reduction; do
  for j in 1 2 64; do
    export GOMP_CPU_AFFINITY="0-$(($j - 1))"
    export OMP_NUM_THREADS=$j
    ./$i -s 256 -n 1 -c 4 -l 10
  done
done
