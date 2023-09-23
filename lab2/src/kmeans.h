#pragma once

#include "args.h"

#include <memory>

void
kmeans_sequential(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels,
  int *num_iters
);

inline double vect_sq_dist(double *a, double *b, int dim) {
  double dist = 0;
  double diff = 0;

  for (int i = 0; i < dim; i++) {
    diff = a[i] - b[i];
    dist += diff * diff;
  }

  return dist;
}

template <typename T>
inline void vect_clear(T *a, int dim) {
  for (int i = 0; i < dim; i++) {
    a[i] = 0;
  }
}

template <typename T>
inline void vect_add(T *a, T *b, int dim) {
  for (int i = 0; i < dim; i++) {
    a[i] += b[i];
  }
}

inline void vect_div(double *a, double div, int dim) {
  for (int i = 0; i < dim; i++) {
    a[i] /= div;
  }
}

inline void vect_print(double *a, int dim) {
  printf("[");
  for (int i = 0; i < dim - 1; i++) {
    printf("%0.5f ", a[i]);
  }
  printf("%0.5f]\n", a[dim - 1]);
}
