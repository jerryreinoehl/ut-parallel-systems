#pragma once

#include <string>

class KmeansArgs {
  public:
    KmeansArgs(int argc, char *argv[]);

    bool output_centroids_labels = false; // Output centroids if true and labels if false.
    int num_clusters = 2;
    int num_dims = 10;
    int max_iters = 150;
    unsigned int seed = 8675309;
    double threshold = 1e-5;
    std::string input_file;
    std::string impl;
};
