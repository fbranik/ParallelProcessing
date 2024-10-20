#include <stdio.h>
#include <stdlib.h>

#include "alloc.h"
#include "error.h"
#include "kmeans.h"

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
__host__ __device__ inline static float euclid_dist_2_transpose(int numCoords,
                                                                int numObjs,
                                                                int numClusters,
                                                                float *objects, // [numCoords][numObjs]
                                                                float *clusters,// [numCoords][numClusters]
                                                                int objectId,
                                                                int clusterId) {
  int i;
  float ans = 0.0;
  for (i = 0; i < numCoords; i++) {
    ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
           (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
  }

  return (ans);
}

__global__ static void reduce_newClusterData_and_update_centroids(int numCoords, int numClusters, int numClusterBlocks,
                                                                  float *deviceClusters, int *deviceLocalNewClusterSize,
                                                                  float *deviceLocalNewClusters) {
  /*
          Shared Memory Layout for this Kernel

|-numCoords*numClusters*sizeof(float)-|-numClusters*sizeof(int)-|
|------------newClusters--------------|------newClusterSize-----|

*/
  extern __shared__ float sharedMemory[];

  float *shmemNewClusters = &sharedMemory[0];
  int *shmemNewClusterSize = (int *) (sharedMemory + numClusters * numCoords);

  if (threadIdx.x == 0) {
    // Thread 0 of each block sets the shared memory newClusterSize and newClusters to 0.0 (previously done by CPU))
    memset(shmemNewClusterSize, 0, numClusters * sizeof(int));
    memset(shmemNewClusters, 0, numClusters * numCoords * sizeof(float));
  }

  int iCluster = get_tid();
  int iCoord = threadIdx.y + blockIdx.y * blockDim.y;
  int iBlock;
  for (iBlock = 0; iBlock < numClusterBlocks; iBlock++) {
    if (iCluster < numClusters) {
      shmemNewClusterSize[iCluster] += deviceLocalNewClusterSize[iBlock * numClusters + iCluster];
      if (iCoord < numCoords) {
        shmemNewClusters[numClusters * iCoord + iCluster] += deviceLocalNewClusters[iBlock * numClusters * numCoords +
                                                                                    iCoord * numClusters + iCluster];
      }
    }
  }
  __syncthreads();
  // Update Centroids in Parallel
  if (iCluster < numClusters) {
    if (iCoord < numCoords) {
      if (shmemNewClusterSize[iCluster] > 0) {
        deviceClusters[iCoord * numClusters + iCluster] =
                shmemNewClusters[iCoord * numClusters + iCluster] / shmemNewClusterSize[iCluster];
      }
    }
  }
}

__global__ static void find_nearest_cluster(int numCoords,
                                            int numObjs,
                                            int numClusters,
                                            float *objects,       //  [numCoords][numObjs]
                                            float *deviceClusters,//  [numCoords][numClusters]
                                            int *deviceMembership,//  [numObjs]
                                            float *devdelta,
                                            int *deviceLocalNewClusterSize,
                                            float *deviceLocalNewClusters) {
  extern __shared__ float sharedMemory[];
  int i;
  int tid = get_tid();

  /*

   Shared Memory Layout for this Kernel (Each block has this layout and reduction is performed later):
|-numCoords*numClusters*sizeof(float)-|-numCoords*numClusters*sizeof(float)-|-numClusters*sizeof(int)-|
|-----------deviceClusters------------|------------newClusters--------------|------newClusterSize-----|

*/

  float *shmemClusters = &sharedMemory[0];
  float *shmemNewClusters = &sharedMemory[numClusters * numCoords];
  int *shmemNewClusterSize = (int *) (sharedMemory + 2 * numClusters * numCoords);

  if (threadIdx.x < numClusters) {// each thread copies an element of deviceClusters into the shared memory
    for (i = 0; i < numCoords; i++) {
      memcpy(&shmemClusters[i * numClusters + threadIdx.x], &deviceClusters[i * numClusters + threadIdx.x],
             sizeof(float));
    }
  }
  if (threadIdx.x == 0) {
    // Thread 0 of each block sets the shared memory newClusterSize and newClusters to 0.0 (previously done by CPU))
    memset(shmemNewClusterSize, 0, numClusters * sizeof(int));
    memset(shmemNewClusters, 0, numClusters * numCoords * sizeof(float));
  }
  __syncthreads();

  /* Get the global ID of the thread. */

  // this could be also done with a Grid-Stride Loop
  if (tid < numObjs) {
    int index;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index = 0;
    min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, 0);
    for (i = 1; i < numClusters; i++) {
      dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, i);
      /* no need square root */
      if (dist < min_dist) { /* find the min and its array index */
        min_dist = dist;
        index = i;
      }
    }

    if (deviceMembership[tid] != index) {
      atomicAdd(devdelta, 1.0);
    }

    /* assign the deviceMembership to object objectId */
    deviceMembership[tid] = index;

    atomicAdd(&shmemNewClusterSize[index], 1);
    for (i = 0; i < numCoords; i++) {
      atomicAdd(&shmemNewClusters[i * numClusters + index], objects[i * numObjs + tid]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    /*
        Thread 0 of each block's copies the block's shared memory to global memory locations so that reduction can be performed
    */
    memcpy(&deviceLocalNewClusterSize[blockIdx.x * numClusters], shmemNewClusterSize, numClusters * sizeof(int));
    memcpy(&deviceLocalNewClusters[blockIdx.x * numClusters * numCoords], shmemNewClusters,
           numClusters * numCoords * sizeof(float));
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
                int numCoords,       /* no. features */
                int numObjs,         /* no. objects */
                int numClusters,     /* no. clusters */
                float threshold,     /* % objects change membership */
                long loop_threshold, /* maximum number of iterations */
                int *membership,     /* out: [numObjs] */
                float *clusters,     /* out: [numClusters][numCoords] */
                int blockSize) {
  double timing = wtime(), gpuTotalTime = 0.0, cpuTotalTime = 0.0, transfersTotalTime = 0.0, timingGPU, timingTransfers, timingCPU, timing_internal, timer_min = 1e42, timer_max = 0;
  int loop_iterations = 0;
  int i, j, index, loop = 0;
  int *newClusterSize;             /* [numClusters]: no. objects assigned in each
                                new cluster */
  float delta = 0, *dev_delta_ptr; /* % of objects change their clusters */

  float **dimObjects = (float **) calloc_2d(numCoords, numObjs, sizeof(float));     //-> [numCoords][numObjs]
  float **dimClusters = (float **) calloc_2d(numCoords, numClusters, sizeof(float));//-> [numCoords][numClusters]
  float **newClusters = (float **) calloc_2d(numCoords, numClusters, sizeof(float));//-> [numCoords][numClusters]

  float *deviceObjects;
  float *deviceClusters, *deviceLocalNewClusters;
  int *deviceMembership, *deviceLocalNewClusterSize;

  printf("\n|-----------All GPU Kmeans------------|\n\n");

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
  const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize) ? blockSize : numObjs;
  const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

  dim3 numThreadsPerBlockUpdateCentroids(numClusters, numCoords, 1);
  dim3 numBlocksUpdateCentroids(1, 1, 1);

  const unsigned int clusterBlockSharedDataSize =
          2 * numClusters * numCoords * sizeof(float) + numClusters * sizeof(int);
  const unsigned int clusterBlockSharedDataSizeUpdateCentroids =
          numClusters * numCoords * sizeof(float) + numClusters * sizeof(int);

  cudaDeviceProp deviceProp;
  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaGetDeviceProperties(&deviceProp, deviceNum);

  if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock ||
      clusterBlockSharedDataSizeUpdateCentroids > deviceProp.sharedMemPerBlock) {
    error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids %d, %d, %d\n",
          numClusterBlocks, numClusters, numCoords);
  }

  checkCuda(cudaMalloc(&deviceObjects, numObjs * numCoords * sizeof(float)));
  checkCuda(cudaMalloc(&deviceClusters, numClusters * numCoords * sizeof(float)));

  checkCuda(cudaMalloc(&deviceLocalNewClusters, numClusterBlocks * numClusters * numCoords * sizeof(float)));
  checkCuda(cudaMalloc(&deviceLocalNewClusterSize, numClusterBlocks * numClusters * sizeof(int)));

  checkCuda(cudaMalloc(&deviceMembership, numObjs * sizeof(int)));
  checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(float)));

  timing = wtime() - timing;
  printf("t_alloc_gpu: %lf ms\n\n%d, %d, %d\n", 1000 * timing, clusterBlockSharedDataSize, numThreadsPerClusterBlock,
         numClusterBlocks);
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

    checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], sizeof(float) * numClusters * numCoords,
                         cudaMemcpyHostToDevice));

    checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(float)));

    checkCuda(cudaMemset(deviceLocalNewClusters, 0, numClusterBlocks * numClusters * numCoords * sizeof(float)));
    checkCuda(cudaMemset(deviceLocalNewClusterSize, 0, numClusterBlocks * numClusters * sizeof(int)));


    transfersTotalTime += wtime() - timingTransfers;
    timingGPU = wtime();

    //printf("Launching find_nearest_clusterrest_cluster_and_update_centroids Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
    find_nearest_cluster<<<numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize>>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters,
             deviceMembership, dev_delta_ptr,
             deviceLocalNewClusterSize, deviceLocalNewClusters);

    cudaDeviceSynchronize();
    checkLastCudaError();

    reduce_newClusterData_and_update_centroids<<<numBlocksUpdateCentroids, numThreadsPerBlockUpdateCentroids, clusterBlockSharedDataSizeUpdateCentroids>>>
            (numCoords, numClusters, numClusterBlocks,
             deviceClusters, deviceLocalNewClusterSize, deviceLocalNewClusters);

    cudaDeviceSynchronize();
    checkLastCudaError();

    gpuTotalTime += wtime() - timingGPU;
    timingTransfers = wtime();
    //printf("Kernels complete for itter %d, updating data in CPU\n", loop);

    checkCuda(cudaMemcpy(dimClusters[0], deviceClusters, numCoords * numClusters * sizeof(float),
                         cudaMemcpyDeviceToHost));


    checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(float), cudaMemcpyDeviceToHost));
    transfersTotalTime += wtime() - timingTransfers;

    timingCPU = wtime();

    delta /= numObjs;

    loop++;

    cpuTotalTime += wtime() - timingCPU;
    timing_internal = wtime() - timing_internal;
    if (timing_internal < timer_min) timer_min = timing_internal;
    if (timing_internal > timer_max) timer_max = timing_internal;
  } while (delta > threshold && loop < loop_threshold);

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
  fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "Shmem", blockSize, timing / loop, timer_min, timer_max);
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
