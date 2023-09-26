#include "args.h"

#include <string>
#include <iostream>

KmeansArgs::KmeansArgs(int argc, char *argv[]) {
  std::string arg;
  for (char **it = &argv[0], **end = &argv[argc]; it != end; ++it) {
    arg = std::string(*it);

    if (arg == "-c")
      output_centroids_labels = true;
    else if (arg == "-k")
      num_clusters = std::stoi(*(++it));
    else if (arg == "-d")
      num_dims = std::stoi(*(++it));
    else if (arg == "-m")
      max_iters = std::stoi(*(++it));
    else if (arg == "-s")
      seed = std::stoi(*(++it));
    else if (arg == "-t")
      threshold = std::stod(*(++it));
    else if (arg == "-i")
      input_file = *(++it);
    else if (arg == "--impl")
      impl = *(++it);
  }
}
