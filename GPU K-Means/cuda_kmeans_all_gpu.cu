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
    return threadIdx.x + blockDim.x * blockIdx.x; /* TODO: copy me from naive version... */
}

__device__ int get_tidY() {
    return threadIdx.y + blockDim.y * blockIdx.y; /* TODO: copy me from naive version... */
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

    /* TODO: Copy me from transpose version*/

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

    int iBlock = get_tid();
    int iCoord, iCluster, i;
    if (0 < iBlock < numClusterBlocks) {
        for (iCluster = 0; iCluster < numClusters; iCluster++) {
            atomicAdd(&deviceLocalNewClusterSize[iCluster], deviceLocalNewClusterSize[iBlock * numClusters + iCluster]);
            for (iCoord = 0; iCoord < numCoords; iCoord++) {
               atomicAdd(&deviceLocalNewClusters[numClusters * iCoord + iCluster], deviceLocalNewClusters[iBlock * numClusters * numCoords + iCoord * numClusters + iCluster]);
            }
        }
    }
    __syncthreads();
    // Update Centroids in Parallel
    if (iBlock < numClusters) {
        for (i = 0; i < numCoords; i++) {
            if (deviceLocalNewClusterSize[iBlock] > 0) {
                deviceClusters[i * numClusters + iBlock] = deviceLocalNewClusters[i * numClusters + iBlock] / deviceLocalNewClusterSize[iBlock];
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
            memcpy(&shmemClusters[i * numClusters + threadIdx.x], &deviceClusters[i * numClusters + threadIdx.x], sizeof(float));
        }
    }
    if (threadIdx.x == 0) {
        // Thread 0 of each block sets the sharerd memory newClusterSize and newClusters to 0.0 (previoulsly done by CPU))
        memset(shmemNewClusterSize, 0, numClusters * sizeof(int));
        memset(shmemNewClusters, 0, numClusters * numCoords * sizeof(float));
    }
    __threadfence();
    __syncthreads();
    /* TODO: Copy deviceClusters to shmemClusters so they can be accessed faster.
		BEWARE: Make sure operations is complete before any thread continues... */
    /* Get the global ID of the thread. */

    /* TODO: Maybe something is missing here... should all threads run this? */
    if (tid < numObjs) {
        int index;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index = 0;
        /* TODO: call min_dist = euclid_dist_2(...) with correct objectId/clusterId using clusters in shmem*/
        min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, 0);
        for (i = 1; i < numClusters; i++) {
            /* TODO: call dist = euclid_dist_2(...) with correct objectId/clusterId */
            dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index = i;
            }
        }
        //__threadfence();

        if (deviceMembership[tid] != index) {
            /* TODO: Maybe something is missing here... is this write safe? */
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
			      Thread 0 of each blocks copies the block's shared memory to global memory locations so that reduction can be performed
		    */
        memcpy(&deviceLocalNewClusterSize[blockIdx.x * numClusters], shmemNewClusterSize, numClusters * sizeof(int));
        memcpy(&deviceLocalNewClusters[blockIdx.x * numClusters * numCoords], shmemNewClusters, numClusters * numCoords * sizeof(float));
    }
    /*if (threadIdx.x<numClusters){
		deviceLocalNewClusterSize[blockIdx.x*numClusters+threadIdx.x] = shmemNewClusters[threadIdx.x];
		for (i=0; i<numCoords; i++){
			deviceLocalNewClusters[blockIdx.x*numClusters*numCoords+i*numClusters+threadIdx.x] = shmemNewClusters[i*numClusters+threadIdx.x];
		}
	}*/
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
    /* TODO: Copy me from transpose version*/
    float **dimObjects = (float **) calloc_2d(numCoords, numObjs, sizeof(float));     //-> [numCoords][numObjs]
    float **dimClusters = (float **) calloc_2d(numCoords, numClusters, sizeof(float));//-> [numCoords][numClusters]
    float **newClusters = (float **) calloc_2d(numCoords, numClusters, sizeof(float));//-> [numCoords][numClusters]

    float *deviceObjects;
    float *deviceClusters, *deviceLocalNewClusters;
    int *deviceMembership, *deviceLocalNewClusterSize;

    printf("\n|-----------All GPU Kmeans------------|\n\n");

    /* TODO: Copy me from transpose version*/
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
    const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock; /* TODO: Calculate Grid size, e.g. number of blocks. */

    const unsigned int numThreadsPerBlockUpdateCentroids = 1024;
    const unsigned int numBlocksUpdateCentroids   = numClusterBlocks/numThreadsPerBlockUpdateCentroids;

    /*	Define the shared memory needed per block.
    	- BEWARE: We can overrun our shared memory here if there are too many
    	clusters or too many coordinates! 
    	- This can lead to occupancy problems or even inability to run. 
    	- Your exercise implementation is not requested to account for that (e.g. always assume deviceClusters fit in shmemClusters */
    const unsigned int clusterBlockSharedDataSize                = 2 * numClusters * numCoords * sizeof(float) + numClusters * sizeof(int);
    const unsigned int clusterBlockSharedDataSizeUpdateCentroids = numClusters * numCoords * sizeof(float) + numClusters * sizeof(int);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock || clusterBlockSharedDataSizeUpdateCentroids > deviceProp.sharedMemPerBlock) {
        error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids %d, %d, %d\n", numClusterBlocks, numClusters, numCoords);
    }

    checkCuda(cudaMalloc(&deviceObjects, numObjs * numCoords * sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters * numCoords * sizeof(float)));

    checkCuda(cudaMalloc(&deviceLocalNewClusters, numClusterBlocks * numClusters * numCoords * sizeof(float)));
    checkCuda(cudaMalloc(&deviceLocalNewClusterSize, numClusterBlocks * numClusters * sizeof(int)));

    checkCuda(cudaMalloc(&deviceMembership, numObjs * sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(float)));

    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n%d, %d, %d\n", 1000 * timing, clusterBlockSharedDataSize, numThreadsPerClusterBlock, numClusterBlocks);
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

        /* TODO: Copy clusters to deviceClusters
        checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], sizeof(float) * numClusters * numCoords, cudaMemcpyHostToDevice));

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

        /* TODO: Copy deviceMembership to membership
        checkCuda(cudaMemcpy(...)); */
        //		checkCuda(cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));
        //		checkCuda(cudaMemcpy(newClusterSize, deviceNewClusterSize, numClusters*sizeof(int), cudaMemcpyDeviceToHost));
        //		checkCuda(cudaMemcpy(newClusters[0], deviceNewClusters, numCoords*numClusters*sizeof(float), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(dimClusters[0], deviceClusters, numCoords * numClusters * sizeof(float), cudaMemcpyDeviceToHost));

        /* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(float), cudaMemcpyDeviceToHost));
        transfersTotalTime += wtime() - timingTransfers;

        timingCPU = wtime();
        /* CPU part: Update cluster centers*/
        /*
        for (i=0; i<numObjs; i++) {
            // find the array index of nestest cluster center 
            index = membership[i];
			
            // update new cluster centers : sum of objects located within 
        //    newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[j][index] += objects[i*numCoords + j];
        }
 */
        /* average the sum and replace old cluster centers with newClusters */

        /* 		for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
//	               newClusters[j][i] = 0.0;   // set back to 0 
            }
//            newClusterSize[i] = 0;   // set back to 0 
        }
*/
        delta /= numObjs;
        //printf("delta is %f - ", delta);
        loop++;
        /*		printf("delta is %f - ", delta);
        printf("completed loop %d\n", loop);
        fflush(stdout);
         printf("\tcompleted loop %d\n", loop);
         for (i=0; i<numClusters; i++) {
             printf("\tclusters[%ld] = ",i);
             for (j=0; j<numCoords; j++)
                 printf("%6.6f ", dimClusters[j][i]);
             printf("\n");
         }
*/
        cpuTotalTime += wtime() - timingCPU;
        timing_internal = wtime() - timing_internal;
        if (timing_internal < timer_min) timer_min = timing_internal;
        if (timing_internal > timer_max) timer_max = timing_internal;
    } while (delta > threshold && loop < loop_threshold);

    /*TODO: Update clusters using dimClusters. Be carefull of layout!!! clusters[numClusters][numCoords] vs dimClusters[numCoords][numClusters] */
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
    sprintf(outfile_name, "Execution_logs/Sz-%ld_Coo-%d_Cl-%d.csv", numObjs * numCoords * sizeof(float) / (1024 * 1024), numCoords, numClusters);
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
