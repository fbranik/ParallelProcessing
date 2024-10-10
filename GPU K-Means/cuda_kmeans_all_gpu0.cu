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

__device__ int get_tid(){
	return threadIdx.x + blockDim.x*blockIdx.x; /* TODO: copy me from naive version... */
}

/* square of Euclid distance between two multi-dimensional points using column-base format */
__host__ __device__ inline static
float euclid_dist_2_transpose(int numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;
	for(i=0; i<numCoords; i++){
		ans+=(objects[numObjs*i+objectId]-clusters[numClusters*i+clusterId])*
	 		 (objects[numObjs*i+objectId]-clusters[numClusters*i+clusterId]);
	}

	/* TODO: Calculate the euclid_dist of elem=objectId of objects from elem=clusterId from clusters, but for column-base format!!! */

    return(ans);
}

__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *deviceMembership,          //  [numObjs]
                          float *devdelta,
						  int* deviceNewClusterSizes)
{
    /*
		Shared Memory Layout:
		
		|-numCoords*numClusters*sizeof(float)-|-numCoords*numClusters*sizeof(float)-|-numClusters*sizeof(int)-| 
		|-----------deviceClusters------------|------------newClusters--------------|------newClusterSize-----| 
	
	*/
	extern __shared__ float sharedMemory[];
	
	float* shmemClusters    = &sharedMemory[0];
	float* shmemNewClusters = &sharedMemory[numClusters*numCoords];
	int* shmemNewClusterSize = (int*)(sharedMemory+2*numClusters*numCoords);
	
	int i;
    int tid = get_tid(); 
	if(threadIdx.x<numClusters){ // each thread copies an element of deviceClusters into the shared memory
		for(i=0; i<numCoords; i++){
			//Same as shared version, copy deviceClusters to shared memory
			memcpy(&shmemClusters[i*numClusters+threadIdx.x], &deviceClusters[i*numClusters+threadIdx.x], sizeof(float));
		}
	}
	if(tid<numClusters){ // each thread copies an element of deviceClusters into the shared memory

		// Set newClusterSize to 0.0 (previoulsly done by CPU)) 
		memset(&shmemNewClusterSize[tid], 0, sizeof(int));
		for(i=0; i<numCoords; i++){
			//Set the newClusters' elements (in shared memory) to 0.0 
			memset(&shmemNewClusters[i*numClusters+tid], 0.0, sizeof(float));
		}
	}
	 __threadfence();	
	/* TODO: Copy deviceClusters to shmemClusters so they can be accessed faster. 
		BEWARE: Make sure operations is complete before any thread continues... */
	/* Get the global ID of the thread. */

	/* TODO: Maybe something is missing here... should all threads run this? */
    if (tid<numObjs) {
        int   index;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index = 0;
        /* TODO: call min_dist = euclid_dist_2(...) with correct objectId/clusterId using clusters in shmem*/
		min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, 0);
        for (i=1; i<numClusters; i++) {
            /* TODO: call dist = euclid_dist_2(...) with correct objectId/clusterId */
			dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmemClusters, tid, i);            
			/* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (deviceMembership[tid] != index) {
        	/* TODO: Maybe something is missing here... is this write safe? */
			atomicAdd(devdelta, 1.0);
        }

        /* assign the deviceMembership to object objectId */
        deviceMembership[tid] = index;
		//Old CPU part starts here
		atomicAdd(&shmemNewClusterSize[index], 1);
		__threadfence();	
		for(i=0; i<numCoords; i++){
			//newClusters[j][index] += objects[i*numCoords + j];
			atomicAdd(&shmemNewClusters[i*numClusters+index],objects[tid*numCoords+i]);	
		}
		__threadfence();

		if (tid<numClusters){
		memcpy(&deviceClusters[tid], &shmemNewClusters[tid], sizeof(float)*numCoords);//*numCoords);
		memcpy(&deviceNewClusterSizes[tid],&shmemNewClusterSize[tid],sizeof(int));		
		}
	}
//	__threadfence();
//	if(blockIdx.x<numClusters){		
//   		if (shmemNewClusterSize[blockIdx.x] > 0){
////       		for(i=0; i<numCoords; i++){	
//			if(threadIdx.x<numCoords){
//				atomicExch(&deviceClusters[blockIdx.x+threadIdx.x*numClusters], shmemNewClusters[blockIdx.x+threadIdx.x*numClusters] / (float)shmemNewClusterSize[blockIdx.x]);
//       		}
//		}
//	}	

}
/*
__global__ static
void update_centroids(int numCoords,
                          int numClusters,
                          float *deviceClusters)    //  [numCoords][numClusters])
{
	extern __shared__ float sharedMemory[];
	float* shmemClusters    = &sharedMemory[0];
	float* shmemNewClusters = &sharedMemory[numClusters*numCoords];
	int* shmemNewClusterSize = (int*)(sharedMemory+2*numClusters*numCoords);
	
	int i;
    int tid = get_tid(); 
		__threadfence();
			
}
*/
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
void kmeans_gpu(	float *objects,      /* in: [numObjs][numCoords] */
		               	int     numCoords,    /* no. features */
		               	int     numObjs,      /* no. objects */
		               	int     numClusters,  /* no. clusters */
		               	float   threshold,    /* % objects change membership */
		               	long    loop_threshold,   /* maximum number of iterations */
		               	int    *membership,   /* out: [numObjs] */
						float * clusters,   /* out: [numClusters][numCoords] */
						int blockSize)  
{
    double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0; 
	int    loop_iterations = 0; 
    int      i, j, index, loop=0;
    float  delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
    
	float  **dimObjects  = (float**) calloc_2d(numCoords, numObjs, sizeof(float)) 	 ; //-> [numCoords][numObjs]
    float  **dimClusters = (float**) calloc_2d(numCoords, numClusters, sizeof(float)); //-> [numCoords][numClusters]
    float  **newClusters = (float**) calloc_2d(numCoords, numClusters, sizeof(float)); //-> [numCoords][numClusters]
    

    printf("\n|-----------Full-offload GPU Kmeans------------|\n\n");
    
    /* TODO: Copy me from transpose version*/
	for(i=0; i<numCoords; i++){
		for(j=0; j<numObjs; j++){
			memcpy(&dimObjects[i][j],&objects[j*numCoords+i], sizeof(float));
		}
	}
    
    float *deviceObjects;
    float *deviceClusters, *devicenewClusters;
    int *deviceMembership;
    int *newClusterSize, *deviceNewClusterSizes; /* [numClusters]: no. objects assigned in each new cluster */
    
    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }
	
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL); 
    
    timing = wtime() - timing;
    printf("t_alloc: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
    const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize)? blockSize: numObjs;
    const unsigned int numClusterBlocks = (numObjs+numThreadsPerClusterBlock-1)/numThreadsPerClusterBlock; /* TODO: Calculate Grid size, e.g. number of blocks. */

	/*	Define the shared memory needed per block.
    	- BEWARE: We can overrun our shared memory here if there are too many
    	clusters or too many coordinates! 
    	- This can lead to occupancy problems or even inability to run. 
    	- Your exercise implementation is not requested to account for that (e.g. always assume deviceClusters fit in shmemClusters */
    const unsigned int clusterBlockSharedDataSize = 2*numClusters*numCoords*sizeof(float)+numClusters*sizeof(int);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids\n");
    }
           
    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&devicenewClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceNewClusterSizes, numClusters*sizeof(int)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(float)));
 
    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
       
    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(deviceNewClusterSizes, 0, numClusters*sizeof(int)));
    free(dimObjects[0]);
      
    timing = wtime() - timing;
    printf("t_get_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();   
    
    do {
        timing_internal = wtime(); 
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], sizeof(float)*numClusters*numCoords, cudaMemcpyHostToDevice)); 

		checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(float)));          
		//printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
        /* TODO: change invocation if extra parameters needed 
        */
		find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, dev_delta_ptr, deviceNewClusterSizes);
        

        cudaDeviceSynchronize(); checkLastCudaError();
		//printf("Kernels complete for itter %d, updating data in CPU\n", loop);
    
    	/* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
		
     	//const unsigned int update_centroids_block_sz = (numCoords* numClusters > blockSize) ? blockSize: numCoords* numClusters;  // TODO: can use different blocksize here if deemed better 
     	//const unsigned int update_centroids_dim_sz =  -1; //TODO: calculate dim for "update_centroids" and fire it 
     /*	
		update_centroids<<< 1, numClusters, clusterBlockSharedDataSize >>>
            (numCoords, numClusters, deviceClusters);  
        cudaDeviceSynchronize(); checkLastCudaError();   
  	*/	
  		checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(float), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(newClusters[0],deviceClusters,  sizeof(float)*numClusters*numCoords, cudaMemcpyDeviceToHost)); 
		checkCuda(cudaMemcpy(&newClusterSize[0], deviceNewClusterSizes, numClusters*sizeof(int), cudaMemcpyDeviceToHost));
        
        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
		

		delta /= numObjs;
       	//printf("delta is %f - ", delta);
        loop++; 
		printf("delta is %f - ", delta);
        printf("completed loop %d\n", loop);
        fflush(stdout);
         printf("\tcompleted loop %d\n", loop);
         for (i=0; i<numClusters; i++) {
             printf("\tclusters[%ld] = ",i);
             for (j=0; j<numCoords; j++)
                 printf("%6.6f ", dimClusters[j][i]);
             printf("\n");
         }
		timing_internal = wtime() - timing_internal; 
		if ( timing_internal < timer_min) timer_min = timing_internal; 
		if ( timing_internal > timer_max) timer_max = timing_internal; 
	} while (delta > threshold && loop < loop_threshold);
                  	
    checkCuda(cudaMemcpy(membership, deviceMembership,
                 numObjs*sizeof(int), cudaMemcpyDeviceToHost));     
    checkCuda(cudaMemcpy(dimClusters[0], deviceClusters,
                 numClusters*numCoords*sizeof(float), cudaMemcpyDeviceToHost));  
                                   
	for (i=0; i<numClusters; i++) {
		for (j=0; j<numCoords; j++) {
		    clusters[i*numCoords + j] = dimClusters[j][i];
		}
	}
	
    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/Sz-%ld_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(float)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name); 
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "All_GPU", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(devicenewClusters));
    checkCuda(cudaFree(deviceNewClusterSizes));
    checkCuda(cudaFree(deviceMembership));
    free(newClusterSize);

    return;
}

