#include "kmeans_cuda.h"
#include "args.h"

#include <memory>
#include <stdio.h>

#include "kmeans.h" // REMOVE TODO

void kmeans_cuda(
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

  auto d_centroids = cudaptr<double>::make_from(centroids, num_clusters * dim);
  auto d_points = cudaptr<double>::make_from(points, num_points * dim);
  auto d_labels = cudaptr<int>::make_from(labels, num_points);

  auto d_counts = cudaptr<int>::make(num_points);
  auto d_centroids_prev = cudaptr<double>::make(num_clusters * dim);

  kmeans_cuda_kernel<<<8, 256>>>(
    dim,
    num_points,
    d_centroids.get(),
    d_points.get(),
    d_labels.get(),
    d_counts.get()
  );

  cudaDeviceSynchronize();

  d_centroids.to_host(centroids);
  d_points.to_host(points);
  d_labels.to_host(labels);
}

__global__ void kmeans_cuda_kernel(
    int dim,
    int num_points,
    double *centroids,
    double *points,
    int *labels,
    int *counts
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int span = blockDim.x;

  // Reset point counts for each centroid.
  cuda_vect_clear(counts + idx, span);
}
