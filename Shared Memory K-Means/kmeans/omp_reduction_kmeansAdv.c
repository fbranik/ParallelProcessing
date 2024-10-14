#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"
#include <omp.h>

// square of Euclid distance between two multi-dimensional points
inline static float euclid_dist_2(int numdims,  /* no. dimensions */
                                  float *coord1,   /* [numdims] */
                                  float *coord2)   /* [numdims] */
{
  int i;
  float ans = 0.0;

  for (i = 0; i < numdims; i++)
    ans += (coord1[i] - coord2[i]) * (coord1[i] - coord2[i]);

  return ans;
}

inline static int find_nearest_cluster(int numClusters, /* no. clusters */
                                       int numCoords,   /* no. coordinates */
                                       float *object,      /* [numCoords] */
                                       float *clusters)    /* [numClusters][numCoords] */
{
  int index, i;
  float dist, min_dist;

  // find the cluster id that has min distance to object
  index = 0;
  min_dist = euclid_dist_2(numCoords, object, clusters);

  for (i = 1; i < numClusters; i++) {
    dist = euclid_dist_2(numCoords, object, &clusters[i * numCoords]);
    // no need square root
    if (dist < min_dist) { // find the min and its array index
      min_dist = dist;
      index = i;
    }
  }
  return index;
}

void kmeans(float *objects,          /* in: [numObjs][numCoords] */
            int numCoords,        /* no. coordinates */
            int numObjs,          /* no. objects */
            int numClusters,      /* no. clusters */
            float threshold,        /* minimum fraction of objects that change membership */
            long loop_threshold,   /* maximum number of iterations */
            int *membership,       /* out: [numObjs] */
            float *clusters)         /* out: [numClusters][numCoords] */
{
  int i, j, k;
  int index, loop = 0;
  double timing = 0;

  float delta;          // fraction of objects whose clusters change in each loop
  int *newClusterSize; // [numClusters]: no. objects assigned in each new cluster
  float *newClusters;  // [numClusters][numCoords]
  int nthreads;         // no. threads

  nthreads = omp_get_max_threads();
  printf("OpenMP Kmeans - Reduction\t(number of threads: %d)\n", nthreads);

  // initialize membership
  for (i = 0; i < numObjs; i++)
    membership[i] = -1;

  // initialize newClusterSize and newClusters to all 0
  newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
  newClusters = (typeof(newClusters)) calloc(numClusters * numCoords, sizeof(*newClusters));

  // Each thread calculates new centers using a private space. After that, thread 0 does an array reduction on them.
  int *local_newClusterSize[nthreads];  // [nthreads][numClusters]
  float *local_newClusters[nthreads];   // [nthreads][numClusters][numCoords]

  // Initialize local (per-thread) arrays (and later collect result on global arrays)
  #pragma omp parallel
  {
    int numObjsPerThread = numObjs / omp_get_max_threads();
    // calloc in parallel per thread to take advantage of the first-touch policy
    int thisThreadId = omp_get_thread_num();
    local_newClusterSize[thisThreadId] = (typeof(*local_newClusterSize)) calloc(numClusters,
                                                                                sizeof(**local_newClusterSize));
    local_newClusters[thisThreadId] = (typeof(*local_newClusters)) calloc(numClusters * numCoords,
                                                                          sizeof(**local_newClusters));
  }


  timing = wtime();
  do {
    // before each loop, set cluster data to 0

    for (i = 0; i < numClusters; i++) {
      for (j = 0; j < numCoords; j++)
        newClusters[i * numCoords + j] = 0.0;
      newClusterSize[i] = 0;
    }

    delta = 0.0;

    #pragma omp parallel shared(objects, clusters, membership, local_newClusters, local_newClusterSize)
    {
      int thisThreadId = omp_get_thread_num();
      for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numCoords; j++) {
          local_newClusters[thisThreadId][i * numCoords + j] = 0.0;
        }
        local_newClusterSize[thisThreadId][i] = 0;
      }

      #pragma omp for private(i, j, index) reduction(+:delta) schedule(static, numObjsPerThread)
      for (i = 0; i < numObjs; i++) {
        // find the array index of nearest cluster center
        index = find_nearest_cluster(numClusters, numCoords, &objects[i * numCoords], clusters);

        // if membership changes, increase delta by 1
        if (membership[i] != index)
          delta += 1.0;

        // assign the membership to object i
        membership[i] = index;

        // update new cluster centers : sum of all objects located within (average will be performed later)
        /*
         * Collect cluster data in local arrays (local to each thread)
         * Replace global arrays with local per-thread
         */
        local_newClusterSize[thisThreadId][index]++;
        for (j = 0; j < numCoords; j++)
          local_newClusters[thisThreadId][index * numCoords + j] += objects[i * numCoords + j];
      }
    }

    /*
     * Reduction of cluster data from local arrays to shared.
     * This operation will be performed by one thread
     */
    int iThread;
    for (i = 0; i < numClusters; i++) {
      for (iThread = 0; iThread < nthreads; iThread++) {
        newClusterSize[i] += local_newClusterSize[iThread][i];
        local_newClusterSize[iThread][i] = 0;
        for (j = 0; j < numCoords; j++) {
          newClusters[i * numCoords + j] += local_newClusters[iThread][i * numCoords + j];
          local_newClusters[iThread][i * numCoords + j] = 0.0;
        }
      }
    }
    // average the sum and replace old cluster centers with newClusters
    for (i = 0; i < numClusters; i++) {
      for (j = 0; j < numCoords; j++) {
        if (newClusterSize[i] > 1)
          clusters[i * numCoords + j] = newClusters[i * numCoords + j] / newClusterSize[i];
      }
    }

    // Get fraction of objects whose membership changed during this loop. This is used as a convergence criterion.
    delta /= numObjs;

    loop++;
    printf("\r\tcompleted loop %d (%f, %f)", loop, delta, threshold);
    fflush(stdout);
  } while (delta > threshold && loop < loop_threshold);
  timing = wtime() - timing;
  printf("\n        nloops = %3d   (total = %7.4fs)  (per loop = %7.4fs)\n", loop, timing, timing / loop);

  for (k = 0; k < nthreads; k++) {
    free(local_newClusterSize[k]);
    free(local_newClusters[k]);
  }
  free(newClusters);
  free(newClusterSize);
}
