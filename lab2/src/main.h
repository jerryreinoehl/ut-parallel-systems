#pragma once

#include <string>
#include <vector>
#include <memory>

std::unique_ptr<double[]> read_points(const std::string& filename, int dim, int *num_points);

int kmeans_rand();
void kmeans_srand(unsigned int seed);
