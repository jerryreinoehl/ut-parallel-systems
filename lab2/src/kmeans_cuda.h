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
    // Allocate memory on the device of size `size * sizeof(T)`.
    cudaptr static make(size_t size) {
      T *ptr;

      cudaError_t err = cudaMalloc(&ptr, size * sizeof(T));

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }

      return {ptr, size};
    }

    // Allocate memory on the device of size `size * sizeof(T)` and initialize
    // with memory from host pointer `*data`.
    cudaptr static make_from(T *data, size_t size) {
      auto ptr = cudaptr::make(size);
      cudaError_t err = cudaMemcpy(
        ptr.get(), data, size * sizeof(T), cudaMemcpyHostToDevice
      );

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }

      return ptr;
    }

    // Allocate memory on the device of size `size * sizeof(T)` and initialize
    // with memory from host pointer `ptr.get()`.
    cudaptr static make_from(const std::unique_ptr<T[]>& ptr, size_t size) {
      return cudaptr::make_from(ptr.get(), size);
    }

    cudaptr(T *t, size_t size) : data_(t), size_(size) {};

    ~cudaptr() {
      cudaFree(data_);
    };

    // Returns the underlying pointer to device memory.
    T *get() {
      return data_;
    }

    // Copy device memory to host memory. `*host` should point to a region
    // with at least the same size of memory allocated by this cudaptr.
    void to_host(T *host) const {
      cudaError_t err = cudaMemcpy(
        host, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost
      );

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }
    }

    // Copy device memory to host memory. `ptr` should point to a region
    // with at least the same size of memory allocated by this cudaptr.
    void to_host(std::unique_ptr<T[]>& ptr) const {
      to_host(ptr.get());
    }

    // Zeroize this block of memory.
    void zero() {
      cudaError_t err = cudaMemset(data_, 0, size_ * sizeof(T));

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }
    }

  private:
    T *data_;
    size_t size_;
};
