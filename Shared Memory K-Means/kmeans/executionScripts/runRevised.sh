#!/bin/bash

## Give the Job a descriptive name
#PBS -N runNaive

## Output and error files
#PBS -o kmeansRevisedNaiveAndSimpleReduction.out
#PBS -e kmeansRevised.err

## How many machines should we get?
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=05:00:00

## Start
## Run make in the src folder (modify properly)

## export OMP_DYNAMIC=TRUE
module load openmp
cd /home/parallel/parlab07/a21Revised
for i in kmeans_omp_naive kmeans_omp_reduction; do
  for j in 1 2 4 8 16 32 64; do
    export GOMP_CPU_AFFINITY="0-$(($j - 1))"
    export OMP_NUM_THREADS=$j
    ./$i -s 256 -n 16 -c 16 -l 10
  done
done
