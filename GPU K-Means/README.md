# Shared Memory K-Means

Parallel implementations of the K-Means algorithm, designed for a shared memory system. The experiments for this
exercise were facilitated on 4 8-core NUMA nodes (32 physical cores and up to 64 threads).

| Directory | Comment                                                 |
|-----------|---------------------------------------------------------|
| kmeans    | Code for the different versions of the parallel K-Means |
| plots     | Plots from the various experiments                      |

There are 4 different versions of the K-Means algorithm included in the _kmeans_ folder, along with auxiliary functions
and
tools. The parallel versions were implemented using OpenMP.
The following table highlights the differences between the included versions.

| File                                      | Comment                                                                            |
|-------------------------------------------|------------------------------------------------------------------------------------|
| cuda_kmeans_naive.cu                      | Naive approach for GPU parallelization                                             |
| cuda_kmeans_transpose.cu                  | Slightly better approach using transpose coordinates for more efficient memory use |
| cuda_kmeans_shared.cu                     | First implementation using the GPU's shared memory                                 |
| cuda_kmeans_all_gpuSharedMemory2D.cu      | A more advanced version that updates the centroids in parallel with reduction      |
| cuda_kmeans_all_gpuReduceParallelBlock.cu | Slight improvement on the  cuda_kmeans_all_gpuSharedMemory2D version               |

The sequential code-basis for these parallel implementations is identical to the one shown
in the [Shared Memory K-Means](../Shared%20Memory%20K-Means/README.md) directory.

### [Naive GPU kmeans](kmeans/cuda_kmeans_naive.cu)

As a first approach to parellization, 

```c
// Initialize local (per-thread) arrays (and later collect result on global arrays)
#pragma
omp parallel
{
int numObjsPerThread = numObjs / omp_get_max_threads();
// calloc in parallel per thread to take advantage of the first-touch policy
int thisThreadId = omp_get_thread_num();
local_newClusterSize[thisThreadId] = (typeof(*local_newClusterSize)) calloc(numClusters,
sizeof(**local_newClusterSize));
local_newClusters[thisThreadId] = (typeof(*local_newClusters)) calloc(numClusters * numCoords,
sizeof(**local_newClusters));
}

...

#pragma
omp parallel shared(objects, clusters, membership, local_newClusters, local_newClusterSize)
{
int thisThreadId = omp_get_thread_num();
for (i = 0; i < numClusters; i++) {
for (j = 0; j < numCoords; j++) {
local_newClusters[thisThreadId][i * numCoords + j] = 0.0;
}
local_newClusterSize[thisThreadId][i] = 0;
}

#pragma
omp for
private(i, j, index) reduction(+:delta) schedule(static, numObjsPerThread)
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
```

This technique, takes advantage of the first-touch policy, which dictates that a memory page is 'actually' allocated
when
it's first used (read or write operation) by a thread or a process. In this case, by having each thread 'first touch'
its data, the system maps it near the NUMA node associated with the thread. Apart from the advantage of each thread
having the data it's going to use in a close memory location, the above minimizes the negatives of a possible
false-sharing scenario. In this case (which occurs for small data sizes), the data of multiple threads can be stored
in the same memory (cache) line. This can cause the false impression that multiple threads are trying to access the
same data, when concurrent write operations are happening, stalling execution. When the thread-local data are stored
in remote memory locations in the NUMA memory system, this phenomenon can result in great performance loss. A practical
example of this will be shown further bellow.

On the other hand, another data-locality that can be implemented has to do with the initialization of the _objects_
array, which represents the N-dimensional space of the problem. Following the reasoning mentioned above, each thread can
initialize and 'touch' the segment of the array
that it's going to use. To achieve this and to ensure that the access to _objects_ is done in a predefined order by the
threads, static scheduling is used. Code-wise these changes were made in _[file_io.c](kmeans/file_io.c)_:

```c
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
```

### Time Measurements and Plots

#### Execution Times

The following plot shows the execution time for the different parallel implementations. Starting with the naive versions
it is apparent that binding the threads with a specific physical core (using the GOMP_CPU_AFFINITY env. variable) vastly
improves performance. Moving on to the versions that include reduction, when compared to the naive version, performance
is generally better. However, for smaller values of the parameters, there is a huge performance penalty (comparing the
orange and green bars). This can be attributed to false-sharing, as mentioned earlier. This hypothesis is also
backed by the fact that the implementation with reduction and thread-allocated local data (pink bar),
eliminates this effect. Lastly, the implementations with NUMA-aware object initialization introduce a further
improvement
in performance, especially for greater numbers of threads. This is expected, considering the previous explanations.

<img src="plots/allVersionsBars.png" width="1381">
