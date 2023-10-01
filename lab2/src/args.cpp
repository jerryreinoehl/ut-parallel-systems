#include "args.h"

#include <string>
#include <string.h>
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
    else if (arg == "-t") {
      char *threshold_str = *(++it);
      char *end;

      threshold = strtod(threshold_str, &end);
      if (static_cast<size_t>(end - threshold_str) < strlen(threshold_str))
        threshold = 1e-5;
    } else if (arg == "-i")
      input_file = *(++it);
    else if (arg == "--impl")
      impl = *(++it);
  }
}
