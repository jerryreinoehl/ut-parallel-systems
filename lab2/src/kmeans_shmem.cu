#include "kmeans_shmem.cuh"
#include "args.h"

#include <memory>
#include <stdio.h>
#include "cudaptr.h"
#include "cudaarena.h"

#include "kmeans.h" // REMOVE TODO

void kmeans_shmem(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels,
  int *num_iters,
  double *time_ms
) {
  int num_clusters = args.num_clusters;
  int dim = args.num_dims;
  double threshold_sq = args.threshold * args.threshold;
  int converged = 0;
  int iters = 0;

  size_t doubles = (num_clusters + num_clusters + num_points) * dim * sizeof(double);
  size_t ints = (num_points + num_clusters + 1) * sizeof(int);

  CudaArena arena{doubles + ints};

  auto d_centroids = cudaptr<double>::make_from(arena, centroids, num_clusters * dim);
  auto d_points = cudaptr<double>::make_from(arena, points, num_points * dim);
  auto d_labels = cudaptr<int>::make_from(arena, labels, num_points);

  auto d_counts = cudaptr<int>::make(arena, num_clusters);
  auto d_centroids_prev = cudaptr<double>::make(arena, num_clusters * dim);

  auto d_converged = cudaptr<int>::make(arena, 1);

  const int blk = 64;
  int centroid_components = num_clusters * dim;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  while (converged != num_clusters && iters < args.max_iters) {
    iters++;

    d_converged.zero();
    d_counts.zero();
    d_centroids.copy_to(d_centroids_prev);
    d_centroids.zero();

    kmeans_shmem_label<<<(num_points + blk - 1)/blk, blk>>>(
      dim,
      num_points,
      num_clusters,
      d_centroids.get(),
      d_centroids_prev.get(),
      d_points.get(),
      d_labels.get(),
      d_counts.get()
    );

    kmeans_shmem_div_centroids_by_count<<<(centroid_components + blk - 1)/blk, blk>>>(
      d_centroids.get(), d_counts.get(), num_clusters, dim
    );

    kmeans_shmem_check_convergence<<<(num_clusters + blk - 1)/num_clusters, blk>>>(
      d_centroids.get(),
      d_centroids_prev.get(),
      num_clusters,
      threshold_sq,
      d_converged.get(),
      dim
    );

    cudaDeviceSynchronize();

    d_converged.to_host(&converged);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float time;
  cudaEventElapsedTime(&time, start, end);
  *time_ms = time;

  d_centroids.to_host(centroids);
  d_points.to_host(points);
  d_labels.to_host(labels);

  *num_iters = iters;

  cudaEventDestroy(start);
  cudaEventDestroy(end);
}

__global__ void kmeans_shmem_label(
    int dim,
    int num_points,
    int num_clusters,
    double *centroids,
    double *centroids_prev,
    double *points,
    int *labels,
    int *counts
) {
  int pnt = blockIdx.x * blockDim.x + threadIdx.x;

  if (pnt >= num_points)
    return;

  int cent, min_cent;
  double dist, min_dist;

  // Calculate closest centroid for each point and number of points mapped
  // to each centroid.
  min_dist = shmem_vect_sq_dist(&points[pnt * dim], &centroids_prev[0], dim);
  min_cent = 0;

  for (cent = 1; cent < num_clusters; cent++) {
    dist = shmem_vect_sq_dist(&points[pnt * dim], &centroids_prev[cent * dim], dim);
    if (dist < min_dist) {
      min_dist = dist;
      min_cent = cent;
    }
  }

  labels[pnt] = min_cent;
  atomicAdd(&counts[min_cent], 1);

  shmem_vect_atomic_add(&centroids[min_cent * dim], &points[pnt * dim], dim);
}

__global__ void kmeans_shmem_div_centroids_by_count(
  double *centroids, int *counts, int num_clusters, int dim
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int cent = idx / dim;

  if (cent >= num_clusters)
    return;

  if (counts[cent] != 0)
    centroids[idx] /= counts[cent];
}

__global__ void kmeans_shmem_check_convergence(
  double *centroids,
  double *centroids_prev,
  int num_clusters,
  double threshold_sq,
  int *converged,
  int dim
) {
  double dist;
  int cent = blockIdx.x * blockDim.x + threadIdx.x;

  if (cent >= num_clusters)
    return;

  dist = shmem_vect_sq_dist(&centroids[cent * dim], &centroids_prev[cent * dim], dim);
  if (dist < threshold_sq)
    atomicAdd(converged, 1);
}

__device__ double shmem_vect_sq_dist(double *a, double *b, int dim) {
  double dist = 0;
  double diff = 0;

  for (int i = 0; i < dim; i++) {
    diff = a[i] - b[i];
    dist += diff * diff;
  }

  return dist;
}

__device__ void shmem_vect_atomic_add(double *dest, double *addend, int dim) {
  for (int i = 0; i < dim; i++) {
    atomicAdd(&dest[i], addend[i]);
  }
}

__device__ void shmem_vect_add(double *dest, double *addend, int dim) {
  for (int i = 0; i < dim; i++) {
    dest[i] += addend[i];
  }
}
