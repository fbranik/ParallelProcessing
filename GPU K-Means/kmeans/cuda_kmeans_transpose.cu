#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#include "alloc.h"
#include "error.h"

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__device__ int get_tid() {
  return threadIdx.x + blockDim.x * blockIdx.x;
}

/* square of Euclid distance between two multi-dimensional points using column-base format */
__host__ __device__ inline static
float euclid_dist_2_transpose(int numCoords,
                              int numObjs,
                              int numClusters,
                              float *objects,     // [numCoords][numObjs]
                              float *clusters,    // [numCoords][numClusters]
                              int objectId,
                              int clusterId) {
  int i;
  float ans = 0.0;
  for (i = 0; i < numCoords; i++) {
    // changed to follow the transpose coordinates
    ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
           (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
  }

  return (ans);
}

__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numObjs][numCoords]
                          float *deviceClusters,    //  [numClusters][numCoords]
                          int *deviceMembership,          //  [numObjs]
                          float *devdelta) {

  /* Get the global ID of the thread. */
  int tid = get_tid();
  // this could be also done with a Grid-Stride Loop
  if (tid < numObjs) {
    int index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index = 0;
    min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, deviceClusters, tid, 0);
    for (i = 1; i < numClusters; i++) {
      dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, deviceClusters, tid, i);

      /* no need square root */
      if (dist < min_dist) { /* find the min and its array index */
        min_dist = dist;
        index = i;
      }
    }

    if (deviceMembership[tid] != index) {
      //(*devdelta)+= 1.0;
      atomicAdd(devdelta, 1.0);
    }

    /* assign the deviceMembership to object objectId */
    deviceMembership[tid] = index;
  }
}


//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
void kmeans_gpu(float *objects,      /* in: [numObjs][numCoords] */
                int numCoords,    /* no. features */
                int numObjs,      /* no. objects */
                int numClusters,  /* no. clusters */
                float threshold,    /* % objects change membership */
                long loop_threshold,   /* maximum number of iterations */
                int *membership,   /* out: [numObjs] */
                float *clusters,   /* out: [numClusters][numCoords] */
                int blockSize) {
  double timing = wtime(), gpuTotalTime = 0.0, cpuTotalTime = 0.0, transfersTotalTime = 0.0, timingGPU, timingTransfers, timingCPU, timing_internal, timer_min = 1e42, timer_max = 0;
  int loop_iterations = 0;
  int i, j, index, loop = 0;
  int *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
  float delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */

  /* Transpose dims */
  float **dimObjects = (float **) calloc_2d(numCoords, numObjs, sizeof(float)); //-> [numCoords][numObjs]
  float **dimClusters = (float **) calloc_2d(numCoords, numClusters, sizeof(float)); //-> [numCoords][numClusters]
  float **newClusters = (float **) calloc_2d(numCoords, numClusters, sizeof(float)); //-> [numCoords][numClusters]

  float *deviceObjects;
  float *deviceClusters;
  int *deviceMembership;

  printf("\n|-----------Transpose GPU Kmeans (Adv. Timers)------------|\n\n");

  // Copy objects given in [numObjs][numCoords] layout to new [numCoords][numObjs] layout
  for (i = 0; i < numCoords; i++) {
    for (j = 0; j < numObjs; j++) {
      memcpy(&dimObjects[i][j], &objects[j * numCoords + i], sizeof(float));
    }
  }

  /* pick first numClusters elements of objects[] as initial cluster centers*/
  for (i = 0; i < numCoords; i++) {
    for (j = 0; j < numClusters; j++) {
      dimClusters[i][j] = dimObjects[i][j];
    }
  }

  /* initialize membership[] */
  for (i = 0; i < numObjs; i++) membership[i] = -1;

  /* need to initialize newClusterSize and newClusters[0] to all 0 */
  newClusterSize = (int *) calloc(numClusters, sizeof(int));
  assert(newClusterSize != NULL);

  timing = wtime() - timing;
  printf("t_alloc: %lf ms\n\n", 1000 * timing);
  timing = wtime();

  // similar to naive version
  const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize) ? blockSize : numObjs;
  const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) /
                                        numThreadsPerClusterBlock;
  const unsigned int clusterBlockSharedDataSize = 0;

  checkCuda(cudaMalloc(&deviceObjects, numObjs * numCoords * sizeof(float)));
  checkCuda(cudaMalloc(&deviceClusters, numClusters * numCoords * sizeof(float)));
  checkCuda(cudaMalloc(&deviceMembership, numObjs * sizeof(int)));
  checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(float)));
  timing = wtime() - timing;
  printf("t_alloc_gpu: %lf ms\n\n", 1000 * timing);
  timing = wtime();

  checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
                       numObjs * numCoords * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(deviceMembership, membership,
                       numObjs * sizeof(int), cudaMemcpyHostToDevice));
  timing = wtime() - timing;
  printf("t_get_gpu: %lf ms\n\n", 1000 * timing);
  timing = wtime();

  do {
    timing_internal = wtime();
    timingTransfers = wtime();
    /* GPU part: calculate new memberships */

    checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], sizeof(float) * numClusters * numCoords,
                         cudaMemcpyHostToDevice));

    checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(float)));
    transfersTotalTime += wtime() - timingTransfers;
    timingGPU = wtime();

    //printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
    find_nearest_cluster
    <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, dev_delta_ptr);

    cudaDeviceSynchronize();
    checkLastCudaError();
    //printf("Kernels complete for itter %d, updating data in CPU\n", loop);
    gpuTotalTime += wtime() - timingGPU;
    timingTransfers = wtime();

    checkCuda(cudaMemcpy(membership, deviceMembership, numObjs * sizeof(int), cudaMemcpyDeviceToHost));

    checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(float), cudaMemcpyDeviceToHost));
    transfersTotalTime += wtime() - timingTransfers;

    timingCPU = wtime();

    /* CPU part: Update cluster centers*/

    for (i = 0; i < numObjs; i++) {
      /* find the array index of nestest cluster center */
      index = membership[i];

      /* update new cluster centers : sum of objects located within */
      newClusterSize[index]++;
      for (j = 0; j < numCoords; j++)
        newClusters[j][index] += objects[i * numCoords + j];
    }

    /* average the sum and replace old cluster centers with newClusters */
    for (i = 0; i < numClusters; i++) {
      for (j = 0; j < numCoords; j++) {
        if (newClusterSize[i] > 0)
          dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
        newClusters[j][i] = 0.0;   /* set back to 0 */
      }
      newClusterSize[i] = 0;   /* set back to 0 */
    }

    delta /= numObjs;
    //printf("delta is %f - ", delta);
    loop++;
    //printf("completed loop %d\n", loop);
    timing_internal = wtime() - timing_internal;
    cpuTotalTime += wtime() - timingCPU;
    if (timing_internal < timer_min) timer_min = timing_internal;
    if (timing_internal > timer_max) timer_max = timing_internal;
  } while (delta > threshold && loop < loop_threshold);

  // store the new values in the original coordinates format in clusters
  for (i = 0; i < numClusters; i++) {
    for (j = 0; j < numCoords; j++) {
      memcpy(&clusters[i * numCoords + j], &dimClusters[j][i], sizeof(float));
      //clusters[i*numCoords+j] = 0.0;//dimClusters[j][i];
    }
  }

  timing = wtime() - timing;
  printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n",
         loop, 1000 * timing, 1000 * timing / loop, 1000 * timer_min, 1000 * timer_max);
  printf("tTransfers=%lf, tGPU=%lf, tCPU=%lf\n\n", transfersTotalTime, gpuTotalTime, cpuTotalTime);
  char outfile_name[1024] = {0};
  sprintf(outfile_name, "Execution_logs/Sz-%ld_Coo-%d_Cl-%d.csv", numObjs * numCoords * sizeof(float) / (1024 * 1024),
          numCoords, numClusters);
  FILE *fp = fopen(outfile_name, "a+");
  if (!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name);
  fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "Transpose", blockSize, timing / loop, timer_min, timer_max);
  fclose(fp);

  checkCuda(cudaFree(deviceObjects));
  checkCuda(cudaFree(deviceClusters));
  checkCuda(cudaFree(deviceMembership));

  free(dimObjects[0]);
  free(dimObjects);
  free(dimClusters[0]);
  free(dimClusters);
  free(newClusters[0]);
  free(newClusters);
  free(newClusterSize);

  return;
}

