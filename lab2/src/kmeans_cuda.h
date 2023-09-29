// vim: ft=cuda
#pragma once

#include "args.h"
#include "cudaptr.h"
#include <memory>
#include <cuda_runtime_api.h>

__global__ void kmeans_label(
    int dim,
    int num_points,
    int num_clusters,
    double *centroids,
    double *centroids_prev,
    double *points,
    int *labels,
    int *counts
);

__global__ void kmeans_div_centroids_by_count(
  double *centroids, int *counts, int num_clusters, int dim
);

__global__ void kmeans_check_convergence(
  double *centroids,
  double *centroids_prev,
  int num_clusters,
  double threshold_sq,
  int *converged,
  int dim
);

__global__ void kmeans_calculate_point_component_distances(
  double *points,
  double *centroids,
  double *components,
  int num_points,
  int num_clusters,
  int dim
);

__global__ void kmeans_sum_component_diffs(
  double *components,
  int num_points,
  int num_clusters,
  int dim
);

__global__ void kmeans_select_labels(
  double *components,
  int *labels,
  int *counts,
  int num_points,
  int num_clusters,
  int dim
);

__global__ void kmeans_calc_new_centroids(
  double *centroids,
  double *points,
  int *labels,
  int num_clusters,
  int num_points,
  int dim
);

void kmeans_cuda(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels,
  int *num_iters,
  double *time_ms
);
