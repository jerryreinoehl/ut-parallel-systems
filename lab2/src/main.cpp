#include "main.h"
#include "args.h"
#include "kmeans.h"
#include "kmeans_cuda.h"
#include "kmeans_shmem.cuh"
#include "kmeans_thrust.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <memory>

#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
  //cudaFreeAsync(0, 0); // Initial CUDA context creation as soon as possible.
  cudaFree(0);           // `cudaFreeAsync` is a better choice but was not found on Codio.

  KmeansArgs args{argc, argv};

  kmeans_srand(args.seed);

  int num_points;
  std::unique_ptr<double[]> points = read_points(args.input_file, args.num_dims, &num_points);

  std::unique_ptr<double[]> centroids = get_centroids(
    args.num_clusters, num_points, args.num_dims, points
  );

  std::unique_ptr<int[]> labels{new int[num_points]};

  int iters;
  double time_ms;

  if (args.impl == "cuda")
    kmeans_cuda(args, num_points, centroids, points, labels, &iters, &time_ms);
  else if (args.impl == "shmem")
    kmeans_shmem(args, num_points, centroids, points, labels, &iters, &time_ms);
  else if (args.impl == "thrust")
    kmeans_thrust(args, num_points, centroids, points, labels, &iters, &time_ms);
  else
    kmeans_sequential(args, num_points, centroids, points, labels, &iters, &time_ms);

  printf("%d,%lf\n", iters, time_ms / iters);

  if (args.output_centroids_labels) {
    print_centroids(centroids.get(), args.num_clusters, args.num_dims);
  } else {
    print_labels(labels.get(), num_points);
  }

  return 0;
}

std::unique_ptr<double[]> read_points(const std::string& filename, int dim, int *num_points) {
  std::ifstream in{filename};
  std::string line;
  std::stringstream ss;

  int idx;

  std::getline(in, line);
  ss = std::stringstream(line);
  ss >> *num_points;

  std::unique_ptr<double[]> points{new double[*num_points * dim]};

  double *it = points.get();

  while (!in.eof()) {
    double num;

    std::getline(in, line);
    if (line == "")
      continue;

    ss = std::stringstream(line);
    ss >> idx;

    for (int i = 0; i < dim; i++) {
      ss >> num;
      *it++ = num;
    }
  }

  return points;
}

std::unique_ptr<double[]>
get_centroids(int num_clusters, int num_points, int dim, const std::unique_ptr<double[]>& points) {
  std::unique_ptr<double[]> clusters{new double[num_clusters * dim]};

  int idx;
  for (int i = 0; i < num_clusters; i++) {
    idx = kmeans_rand() % num_points;
    for (int j = 0; j < dim; j++) {
      clusters[i * dim + j] = points[idx * dim + j];
    }
  }

  return clusters;
}

void print_centroids(double *centroids, int num_centroids, int dim) {
  for (int cent = 0; cent < num_centroids; cent++) {
    printf("%d ", cent);
    for (int d = 0; d < dim; d++) {
      printf("%lf ", centroids[cent * dim + d]);
    }
    printf("\n");
  }
}

void print_labels(int *labels, int num_points) {
  printf("clusters:");
  for (int pnt = 0; pnt < num_points; pnt++) {
    printf(" %d", labels[pnt]);
  }
}

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
  next = next * 1103515245 + 12345;
  return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
  next = seed;
}
