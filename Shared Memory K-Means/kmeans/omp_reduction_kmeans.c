#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"
#include <omp.h>

// square of Euclid distance between two multi-dimensional points
inline static float euclid_dist_2(int    numdims,  /* no. dimensions */
                                 float * coord1,   /* [numdims] */
                                 float * coord2)   /* [numdims] */
{
    int i;
    float ans = 0.0;

    for(i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return ans;
}

inline static int find_nearest_cluster(int     numClusters, /* no. clusters */
                                       int     numCoords,   /* no. coordinates */
                                       float * object,      /* [numCoords] */
                                       float * clusters)    /* [numClusters][numCoords] */
{
    int index, i;
    float dist, min_dist;

    // find the cluster id that has min distance to object 
    index = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters);

    for(i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);
        // no need square root 
        if (dist < min_dist) { // find the min and its array index
            min_dist = dist;
            index    = i;
        }
    }
    return index;
}

void kmeans(float * objects,          /* in: [numObjs][numCoords] */
            int     numCoords,        /* no. coordinates */
            int     numObjs,          /* no. objects */
            int     numClusters,      /* no. clusters */
            float   threshold,        /* minimum fraction of objects that change membership */
            long    loop_threshold,   /* maximum number of iterations */
            int   * membership,       /* out: [numObjs] */
            float * clusters)         /* out: [numClusters][numCoords] */
{
    int i, j, k;
    int index, loop=0;
    double timing = 0;
    		
    float delta;          // fraction of objects whose clusters change in each loop 
    int * newClusterSize; // [numClusters]: no. objects assigned in each new cluster 
    float * newClusters;  // [numClusters][numCoords] 
    int nthreads;         // no. threads 

    nthreads = omp_get_max_threads();
    printf("OpenMP Kmeans - Reduction\t(number of threads: %d)\n", nthreads);
   	int * local_newClusterSize[nthreads];  // [nthreads][numClusters] 
    float * local_newClusters[nthreads];   // [nthreads][numClusters][numCoords]
	
    // initialize membership
    for (i=0; i<numObjs; i++)
        membership[i] = -1;

    // initialize newClusterSize and newClusters to all 0 
    newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
    newClusters = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));

    // Each thread calculates new centers using a private space. After that, thread 0 does an array reduction on them.

    /*
     * Hint for false-sharing
     * This is noticed when numCoords is low (and neighboring local_newClusters exist close to each other).
     * Allocate local cluster data with a "first-touch" policy.
     */
    // Initialize local (per-thread) arrays (and later collect result on global arrays)
    //#pragma omp parallel for
/*	for (k=0; k<nthreads; k++)
	{
		local_newClusterSize[k] = (typeof(*local_newClusterSize)) calloc(numClusters, sizeof(**local_newClusterSize));
		local_newClusters[k] = (typeof(*local_newClusters)) calloc(numClusters * numCoords, sizeof(**local_newClusters));
	}
	*/
 	/*#pragma omp parallel
	{
	    int ithread = omp_get_thread_num();	    
		local_newClusterSize[ithread] = (typeof(*local_newClusterSize)) calloc(numClusters, sizeof(**local_newClusterSize));
		local_newClusters[ithread] = (typeof(*local_newClusters)) calloc(numClusters * numCoords, sizeof(**local_newClusters));
	}*/
    timing = wtime();
    do {
        // before each loop, set cluster data to 0
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++)
                newClusters[i*numCoords + j] = 0.0;
            newClusterSize[i] = 0;
        }

        delta = 0.0;

        /* 
         * TODO: Initiliaze local cluster data to zero (separate for each thread)
         */
           	
 		#pragma omp parallel
		{
		    int ithread = omp_get_thread_num();	    
			local_newClusterSize[ithread] = (typeof(*local_newClusterSize)) calloc(numClusters, sizeof(**local_newClusterSize));
			local_newClusters[ithread] = (typeof(*local_newClusters)) calloc(numClusters * numCoords, sizeof(**local_newClusters));
				
		    for (j=0; j<numClusters; j++) {
	            for (k=0; k<numCoords; k++){
		            local_newClusters[ithread][j*numCoords + k] = 0.0f;
	            }
			    local_newClusterSize[ithread][j] = 0;
	    	}   	
		
			#pragma omp for private(index) reduction(+: delta)
			for (i=0; i<numObjs; i++){		
		 	   	// find the array index of nearest cluster center 
		 	   	index = find_nearest_cluster(numClusters, numCoords, &objects[i*numCoords], clusters);
		 	               
		 	   	// if membership changes, increase delta by 1 
		 	   	if (membership[i] != index)
		 	       delta += 1.0;
	 	               
	 	       	// assign the membership to object i 
	 	       	membership[i] = index;
	 	               
	 	        // update new cluster centers : sum of all objects located within (average will be performed later) 
	 	        /* 
	                 * TODO: Collect cluster data in local arrays (local to each thread)
	 	         *       Replace global arrays with local per-thread
	 	                */
	           	local_newClusterSize[ithread][index]++;
	           	for (j=0; j<numCoords; j++)
	               	local_newClusters[ithread][index*numCoords + j] += objects[i*numCoords + j];
	        }
		}
		
		for (k=0; k<nthreads; k++){
	    	for (i=0; i<numClusters; i++) {
        		for (j=0; j<numCoords; j++){
            		newClusters[i*numCoords + j] +=  local_newClusters[k][i*numCoords + j];
				}
			newClusterSize[i] += local_newClusterSize[k][i];
    		}
		}

        
	// average the sum and replace old cluster centers with newClusters 
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 1)
                    clusters[i*numCoords + j] = newClusters[i*numCoords + j] / newClusterSize[i];
            }
        }
            
        // Get fraction of objects whose membership changed during this loop. This is used as a convergence criterion.
        delta /= numObjs;
        
        loop++;
        printf("\r\tcompleted loop %d", loop);
        fflush(stdout);
    } while (delta > threshold && loop < loop_threshold);
    timing = wtime() - timing;
    printf("\n        nloops = %3d   (total = %7.4fs)  (per loop = %7.4fs)\n", loop, timing, timing/loop);

    for (k=0; k<nthreads; k++)
    {
        free(local_newClusterSize[k]);
        free(local_newClusters[k]);
    }
    free(newClusters);
    free(newClusterSize);
}
