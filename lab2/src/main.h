#pragma once

#include <string>
#include <vector>
#include <memory>

std::unique_ptr<double[]> read_points(const std::string& filename, int dim, int *num_points);

std::unique_ptr<double[]>
get_centroids(int num_clusters, int num_points, int dim, const std::unique_ptr<double[]>& points);

void print_centroids(double *centroids, int num_centroids, int dim);
void print_labels(int *labels, int num_points);

int kmeans_rand();
void kmeans_srand(unsigned int seed);
