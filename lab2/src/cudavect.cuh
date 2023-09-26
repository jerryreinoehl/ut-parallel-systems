#pragma once

#include <cuda_runtime.h>

template <typename T>
__device__ void cuda_vect_clear(T *vect, int dim) {
  for (int i = 0; i < dim; i++)
    vect[i] = 0;
}

__device__ double cuda_vect_sq_dist(double *a, double *b, int dim) {
  double dist = 0;
  double diff = 0;

  for (int i = 0; i < dim; i++) {
    diff = a[i] - b[i];
    dist += diff * diff;
  }

  return dist;
}

__device__ void cuda_vect_atomic_add(double *dest, double *addend, int dim) {
  for (int i = 0; i < dim; i++) {
    atomicAdd(&dest[i], addend[i]);
  }
}

__device__ void cuda_vect_add(double *dest, double *addend, int dim) {
  for (int i = 0; i < dim; i++) {
    dest[i] += addend[i];
  }
}
