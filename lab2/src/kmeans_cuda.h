// vim: ft=cuda
#pragma once

#include "args.h"
#include <memory>
#include <cuda_runtime_api.h>

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

// CUDA smart pointer.
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

    // Allocate memory on the device of size `size * sizeof(T)` from
    // `CudaArena`, `arena`.
    cudaptr static make(CudaArena& arena, size_t size) {
      T *ptr = arena.alloc<T>(size);
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

    // Allocate memory on the device of size `size * sizeof(T)` and initialize
    // with memory from host pointer `*data`.
    cudaptr static make_from(CudaArena& arena, T *data, size_t size) {
      auto ptr = cudaptr::make(arena, size);
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
    cudaptr static make_from(CudaArena& arena, const std::unique_ptr<T[]>& ptr, size_t size) {
      return cudaptr::make_from(arena, ptr.get(), size);
    }

    // Allocate memory on the device of size `size * sizeof(T)` and initialize
    // with memory from host pointer `*data`.
    cudaptr static async_make_from(CudaArena& arena, T *data, size_t size, cudaStream_t stream) {
      auto ptr = cudaptr::make(arena, size);
      cudaError_t err = cudaMemcpyAsync(
        ptr.get(), data, size * sizeof(T), cudaMemcpyHostToDevice, stream
      );

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }

      return ptr;
    }

    // Allocate memory on the device of size `size * sizeof(T)` and initialize
    // with memory from host pointer `ptr.get()`.
    cudaptr static async_make_from(CudaArena& arena, const std::unique_ptr<T[]>& ptr, size_t size, cudaStream_t stream) {
      return cudaptr::async_make_from(arena, ptr.get(), size, stream);
    }

    cudaptr(T *t, size_t size) : data_(t), size_(size) {};
    cudaptr(void *t, size_t size) : data_(static_cast<T*>(t)), size_(size) {};

    ~cudaptr() {
      //cudaFree(data_);
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

    // Copy host memory to device memory.
    void from_host(T *host) {
      cudaError_t err;
      err = cudaMemcpy(data_, host, size_ * sizeof(T), cudaMemcpyHostToDevice);

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }
    }

    // Copy host memory to device memory.
    void from_host(const std::unique_ptr<T[]>& ptr) {
      from_host(ptr.get());
    }

    // Copy device memory to device memory. `ptr` should point to a region
    // with at least the same size of memory allocated by this cudaptr.
    void copy_to(cudaptr<T>& ptr) {
      cudaError_t err;
      err = cudaMemcpy(ptr.get(), data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);

      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
      }
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
