// vim: ft=cuda
#pragma once

#include "args.h"
#include <memory>
#include <cuda_runtime_api.h>

void kmeans_cuda(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels,
  int *num_iters,
  double *time_ms
);

// Simple CUDA smart pointer.
template <typename T>
class cudaptr {
  public:
    cudaptr static make(size_t size) {
      T *ptr;

      cudaError_t err = cudaMalloc(&ptr, size * sizeof(T));

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }

      return {ptr};
    }

    cudaptr(T *t) : data_(t) {};

    ~cudaptr() {
      cudaFree(data_);
    };

    T *get() {
      return data_;
    }

  private:
    T *data_;
};
