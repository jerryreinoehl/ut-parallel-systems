#pragma once

#include "args.h"

#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct kmeans_label_functor {
  kmeans_label_functor(
    double *centroids,
    double *centroids_prev,
    double *points,
    int dim,
    int num_clusters
  ) {
    centroids_ = centroids;
    centroids_prev_ = centroids_prev;
    points_ = points;
    dim_ = dim;
    num_clusters_ = num_clusters;
  }

  __device__ __host__ int operator()(const int& idx);

  double *centroids_;
  double *centroids_prev_;
  double *points_;
  int dim_;
  int num_clusters_;
};

//struct kmeans_cents_functor {
//  kmeans_cents_functor(
//    double *points,
//    double *centroids_prev,
//    double *centroids,
//    int *labels,
//    int *counts,
//    int dim,
//    int num_points,
//    int num_clusters,
//    double threshold_sq
//  ) {
//    points_ = points;
//    centroids_prev_ = centroids_prev;
//    centroids_ = centroids;
//    labels_ = labels;
//    counts_ = counts;
//    dim_= dim;
//    num_points_ = num_points;
//    num_clusters_ = num_clusters;
//    threshold_sq_ = threshold_sq;
//  }
//
//  __device__ __host__ int operator()(const int& cent);
//
//  double *points_;
//  double *centroids_prev_;
//  double *centroids_;
//  int *labels_;
//  int *counts_;
//  int dim_;
//  int num_points_;
//  int num_clusters_;
//  double threshold_sq_;
//};
struct kmeans_cents_functor {
  kmeans_cents_functor(
    double *points,
    double *centroids_prev,
    double *centroids,
    int *labels,
    int *counts,
    int dim,
    int num_points,
    int num_clusters,
    double threshold_sq
  ) {
    points_ = points;
    centroids_prev_ = centroids_prev;
    centroids_ = centroids;
    labels_ = labels;
    counts_ = counts;
    dim_= dim;
    num_points_ = num_points;
    num_clusters_ = num_clusters;
    threshold_sq_ = threshold_sq;
  }

  __device__ __host__ int operator()(const int& cent);

  double *points_;
  double *centroids_prev_;
  double *centroids_;
  int *labels_;
  int *counts_;
  int dim_;
  int num_points_;
  int num_clusters_;
  double threshold_sq_;
};

__host__ __device__ double thrust_vect_sq_dist(
  double *a, double *b, int dim
);

void kmeans_thrust(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels,
  int *num_iters,
  double *time_ms
);

template <typename T>
void copy(thrust::host_vector<T>& dst, const std::unique_ptr<T[]>& src) {
  for (unsigned long i = 0; i < dst.size(); i++)
    dst[i] = src[i];
}

template <typename T>
void copy(std::unique_ptr<T[]>& dst, const thrust::host_vector<T>& src) {
  for (unsigned long i = 0; i < src.size(); i++)
    dst[i] = src[i];
}

template <typename T>
void vect_print(const thrust::host_vector<T>& vect, int dim) {
  printf("[");
  for (int i = 0; i < dim - 1; i++)
    printf("%d ", vect[i]);
  printf("%d]\n", vect[dim - 1]);
}

template <typename T>
T *ptr(thrust::device_vector<T>& vect) {
  return thrust::raw_pointer_cast(vect.data());
}

void update_centroids(
  thrust::device_vector<double>& centroids,
  thrust::device_vector<double>& points,
  thrust::device_vector<int>& labels,
  thrust::device_vector<int>& counts,
  int num_clusters,
  int num_points,
  int dim
);

__global__ void kmeans_thrust_update_centroids(
  double *centroids,
  double *points,
  int *labels,
  int *counts,
  int num_clusters,
  int num_points,
  int dim
);
