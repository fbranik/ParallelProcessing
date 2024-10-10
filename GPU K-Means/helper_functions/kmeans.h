#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

void kmeans(float * objects, int numCoords, int numObjs, int numClusters, float threshold, long loop_threshold, int *membership, float * clusters);

void kmeans_gpu(float * objects, int numCoords, int numObjs, int numClusters, float threshold, long loop_threshold, int *membership, float * clusters, int block_size);

float * dataset_generation(int numObjs, int numCoords);

int check_repeated_clusters(int, int, float*);

double wtime(void);

extern int _debug;

#endif
