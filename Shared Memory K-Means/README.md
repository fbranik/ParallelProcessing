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

| File                         | Comment                                                      |
|------------------------------|--------------------------------------------------------------|
| seq_kmeans.c                 | Serial version used as a base                                |
| omp_naive_kmeans.c           | Naive approach for parallelization                           |
| omp_reduction_kmeansSimple.c | Implementation using local copies and reduction              |
| omp_reduction_kmeansAdv.c    | Implementation with reduction and NUMA-aware initializations |

### [Serial K-Means/Algorithm Overview](kmeans/seq_kmeans.c)

The main computational kernel of the given implementation of the K-Means algorithm, lies in the _kmeans_ function found
in [seq_kmeans.c](kmeans/seq_kmeans.c) and specifically in the following do-while loop:

```c
do {
    // before each loop, set cluster data to 0
    for (i = 0; i < numClusters; i++) {
      for (j = 0; j < numCoords; j++)
        newClusters[i * numCoords + j] = 0.0;
      newClusterSize[i] = 0;
    }

    delta = 0.0;

    for (i = 0; i < numObjs; i++) {
      // find the array index of nearest cluster center
      index = find_nearest_cluster(numClusters, numCoords, &objects[i * numCoords], clusters);

      // if membership changes, increase delta by 1
      if (membership[i] != index)
        delta += 1.0;

      // assign the membership to object i
      membership[i] = index;

      // update new cluster centers : sum of objects located within
      newClusterSize[index]++;
      for (j = 0; j < numCoords; j++)
        newClusters[index * numCoords + j] += objects[i * numCoords + j];
    }

    // average the sum and replace old cluster centers with newClusters
    for (i = 0; i < numClusters; i++) {
      for (j = 0; j < numCoords; j++) {
        if (newClusterSize[i] > 0)
          clusters[i * numCoords + j] = newClusters[i * numCoords + j] / newClusterSize[i];
      }
    }

    // Get fraction of objects whose membership changed during this loop. This is used as a convergence criterion.
    delta /= numObjs;

    loop++;
    printf("\r\tcompleted loop %d", loop);
    fflush(stdout);
  } while (delta > threshold && loop < loop_threshold);
```

The 3 main phases in the above loop are:

+ the 'zeroing' of the intermediate data structures (the first for-loop)
+ the search for new cluster centers (the second for-loop)
+ the updating of the old cluster data with the changes done in the second step

Of the above, the costlier phase is the second one, where every object in the N-dimensional space is examined to
determine if it should continue to belong in its current cluster.

### [Naive K-Means](kmeans/omp_naive_kmeans.c)

As a first approach to the parallelization of the algorithm, the for loop in which the cluster centers are updated
(via the _find_nearest_cluster_ function) is parallelized.

```c
#pragma omp parallel for private(index) reduction(+:delta)
for (i = 0; i < numObjs; i++) {
  // find the array index of nearest cluster center
  index = find_nearest_cluster(numClusters, numCoords, &objects[i * numCoords], clusters);

  // if membership changes, increase delta by 1
  if (membership[i] != index)
    delta += 1.0;

  // assign the membership to object i
  membership[i] = index;

  // update new cluster centers : sum of objects located within, ensure access is atomic to avoid errors
  #pragma omp atomic
  newClusterSize[index]++;
  for (j = 0; j < numCoords; j++)
    #pragma omp atomic
    newClusters[index * numCoords + j] += objects[i * numCoords + j];
}
```

To ensure that the data shared among threads are updated
without errors, atomic operations are used.

### [K-Means with local data copies and reduction](kmeans/omp_reduction_kmeansSimple.c)

As an addition to the naive version and to avoid the cost of atomic operations, local copies of the shared
data are implemented in this version.

```c
#pragma omp for private(i, j, index) reduction(+:delta)
for (i = 0; i < numObjs; i++) {
  // find the array index of nearest cluster center
  index = find_nearest_cluster(numClusters, numCoords, &objects[i * numCoords], clusters);

  // if membership changes, increase delta by 1
  if (membership[i] != index)
    delta += 1.0;

  // assign the membership to object i
  membership[i] = index;

  // update new cluster centers : sum of all objects located within (average will be performed later)
  
  local_newClusterSize[thisThreadId][index]++;
  for (j = 0; j < numCoords; j++)
    local_newClusters[thisThreadId][index * numCoords + j] += objects[i * numCoords + j];
}
```

After the parallel for, the local copies of the data are reduced by one thread, using addition.

```c
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
```

### [K-Means with local data copies, reduction and NUMA-aware initializations](kmeans/omp_reduction_kmeansAdv.c)

There are two ways to improve how data is initialized/used in the above implementation. Firstly, the copies of the
thread-local data can be allocated and initialized by the threads that are going to use them, i.e. allocation and
'zeroing' can happen in the parallel region.

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

### Measurements and Plots

#### Execution Times

<img src="plots/allVersionsBars.png" width="500">