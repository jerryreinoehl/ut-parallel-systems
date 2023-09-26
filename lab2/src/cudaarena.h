#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

class CudaArena {
  public:
    CudaArena(size_t size) : size_(size) {
      cudaError_t err = cudaMalloc(&data_, size_);

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }
    }

    ~CudaArena() {
      cudaFree(data_);
    }

    template <typename T>
    T *alloc(size_t size) {
      T *ptr = static_cast<T*>(data_);
      data_ = (char*)data_ + (size * sizeof(T));
      return ptr;
    }

  private:
    void *data_;
    size_t size_;

};
