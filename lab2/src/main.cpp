#include "main.h"
#include "args.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <memory>

int main(int argc, char *argv[]) {
  KmeansArgs args{argc, argv};

  kmeans_srand(args.seed);

  int num_points;
  std::unique_ptr<double[]> points = read_points(args.input_file, args.num_dims, &num_points);

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

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
  next = next * 1103515245 + 12345;
  return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
  next = seed;
}
