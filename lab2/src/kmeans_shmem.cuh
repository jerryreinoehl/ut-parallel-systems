#pragma once

#include "args.h"
#include <memory>
#include <cuda_runtime_api.h>

void kmeans_shmem(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels,
  int *num_iters,
  double *time_ms
);

__global__ void kmeans_shmem_label(
    int dim,
    int num_points,
    int num_clusters,
    double *centroids,
    double *centroids_prev,
    double *points,
    int *labels,
    int *counts
);

__global__ void kmeans_shmem_div_centroids_by_count(
  double *centroids, int *counts, int num_clusters, int dim
);

__global__ void kmeans_shmem_check_convergence(
  double *centroids,
  double *centroids_prev,
  int num_clusters,
  double threshold_sq,
  int *converged,
  int dim
);

template <typename T>
__device__ void shmem_vect_clear(T *vect, int dim) {
  for (int i = 0; i < dim; i++)
    vect[i] = 0;
}

__device__ double shmem_vect_sq_dist(double *a, double *b, int dim);

__device__ void shmem_vect_atomic_add(double *dest, double *addend, int dim);

__device__ void shmem_vect_add(double *dest, double *addend, int dim);
