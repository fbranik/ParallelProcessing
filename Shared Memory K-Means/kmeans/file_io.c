#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */
#include <omp.h>

#include "kmeans.h"

float *dataset_generation(int numObjs, int numCoords) {
  float *objects = NULL;
  long i, j;
  // Random values that will be generated will be between 0 and 10.
  float val_range = 10;

  int numObjsPerThread = numObjs / omp_get_max_threads();

  /* allocate space for objects[][] and read all objects */
  objects = (typeof(objects)) malloc(numObjs * numCoords * sizeof(*objects));

  #pragma omp parallel for schedule(static, numObjsPerThread)
  for (i = 0; i < numObjs; i++) {
    unsigned int seed = i;
    for (j = 0; j < numCoords; j++) {
      objects[i * numCoords + j] = (rand_r(&seed) / ((float) RAND_MAX)) * val_range;
      if (_debug && i == 0)
        printf("object[i=%ld][j=%ld]=%f\n", i, j, objects[i * numCoords + j]);
    }
  }

  return objects;
}
