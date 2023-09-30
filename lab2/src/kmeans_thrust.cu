#include "kmeans_thrust.cuh"
#include "kmeans.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>

void kmeans_thrust(
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

  // Create thrust host vectors.
  thrust::host_vector<double> h_centroids(num_clusters * dim);
  copy(h_centroids, centroids);

  thrust::host_vector<double> h_points(num_points * dim);
  copy(h_points, points);

  thrust::host_vector<int> h_labels(num_points);
  thrust::host_vector<int> h_converged(num_clusters);

  // Create thrust device vectors.
  thrust::device_vector<double> d_centroids = h_centroids;
  thrust::device_vector<double> d_centroids_prev = h_centroids;
  thrust::device_vector<double> d_points = h_points;
  thrust::device_vector<int> d_labels = h_labels;
  thrust::device_vector<int> d_counts(num_clusters);
  thrust::device_vector<int> d_converged = h_converged;

  thrust::host_vector<int> h_counts(num_clusters);

  thrust::device_vector<int> seq_pnts(num_points);
  thrust::sequence(seq_pnts.begin(), seq_pnts.end());

  thrust::device_vector<int> seq_cents(num_clusters);
  thrust::sequence(seq_cents.begin(), seq_cents.end());

  kmeans_label_functor kmeans_label(ptr(d_centroids), ptr(d_centroids_prev), ptr(d_points), dim, num_clusters);
  kmeans_cents_functor kmeans_cents(
    ptr(d_points), ptr(d_centroids_prev), ptr(d_centroids),
    ptr(d_labels), ptr(d_counts), dim, num_points, num_clusters, threshold_sq
  );

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  while (converged != num_clusters && iters < args.max_iters) {
    iters++;

    d_centroids_prev = d_centroids;
    thrust::fill(d_centroids.begin(), d_centroids.end(), 0);
    thrust::fill(d_counts.begin(), d_counts.end(), 0);

    thrust::transform(seq_pnts.begin(), seq_pnts.end(), d_labels.begin(), kmeans_label);

    update_centroids(d_centroids, d_points, d_labels, d_counts, num_clusters, num_points, dim);

    thrust::transform(seq_cents.begin(), seq_cents.end(), d_converged.begin(), kmeans_cents);

    converged = thrust::reduce(d_converged.begin(), d_converged.end());
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float time;
  cudaEventElapsedTime(&time, start, end);
  *time_ms = time;

  h_centroids = d_centroids;
  h_labels= d_labels;

  copy(centroids, h_centroids);
  copy(labels, h_labels);
   *num_iters = iters;
}

__device__ __host__ int kmeans_label_functor::operator()(const int& pnt) {
  int dim = dim_;

  int cent, min_cent;
  double dist, min_dist;

  min_dist = thrust_vect_sq_dist(&points_[pnt * dim], &centroids_prev_[0], dim);
  min_cent = 0;

  for (cent = 1; cent < num_clusters_; cent++) {
    dist = thrust_vect_sq_dist(&points_[pnt * dim], &centroids_prev_[cent * dim], dim);
    if (dist < min_dist) {
      min_dist = dist;
      min_cent = cent;
    }
  }

  return min_cent;
}

__device__ __host__ int kmeans_cents_functor::operator()(const int& idx) {
  int cent = idx;
  int count = counts_[cent];

  double diff = 0, dist = 0;

  for (int i = 0; i < dim_; i++) {
    centroids_[cent * dim_ + i] /= count;
    diff = centroids_[cent * dim_ + i] - centroids_prev_[cent * dim_ + i];
    dist += diff * diff;
  }

  if (dist < threshold_sq_)
    return 1;
  else
    return 0;
}

__host__ __device__ double thrust_vect_sq_dist(
  double *a, double *b, int dim
) {
  double dist = 0;
  double diff = 0;

  for (int i = 0; i < dim; i++) {
    diff = a[i] - b[i];
    dist += diff * diff;
  }

  return dist;
}

void update_centroids(
  thrust::device_vector<double>& centroids,
  thrust::device_vector<double>& points,
  thrust::device_vector<int>& labels,
  thrust::device_vector<int>& counts,
  int num_clusters,
  int num_points,
  int dim
) {
  int blk = 128;
  kmeans_thrust_update_centroids<<<(num_points + blk - 1)/blk, blk>>>(
    ptr(centroids), ptr(points), ptr(labels), ptr(counts), num_clusters, num_points, dim
  );
  cudaDeviceSynchronize();
}

__global__ void kmeans_thrust_update_centroids(
  double *centroids,
  double *points,
  int *labels,
  int *counts,
  int num_clusters,
  int num_points,
  int dim
) {
  int pnt = blockDim.x * blockIdx.x + threadIdx.x;
  int cent = labels[pnt];


  if (pnt >= num_points)
    return;

  for (int i = 0; i < dim; i++) {
    atomicAdd(&centroids[cent * dim + i], points[pnt * dim + i]);
  }

  atomicAdd(&counts[cent], 1);
}
