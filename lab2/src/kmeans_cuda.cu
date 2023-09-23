#include "kmeans_cuda.h"
#include "args.h"

#include <memory>
#include <stdio.h>

__global__ void kmeans_cuda_kernel(int *a) {
  *a = 32;
}

void kmeans_cuda(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels,
  int *num_iters,
  double *time_ms
) {
  std::unique_ptr<int> a{new int};
  *a = 5;
  int *d_a;

  cudaError_t cuda_err;
  cuda_err = cudaMalloc((void**)&d_a, sizeof(int));
  kmeans_cuda_kernel<<<1, 1>>>(d_a);
  cudaDeviceSynchronize();
  cudaMemcpy(a.get(), d_a, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_a);

  printf("%d\n", *a);
}
