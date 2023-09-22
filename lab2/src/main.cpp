#include "main.h"
#include "args.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

int main(int argc, char *argv[]) {
  KmeansArgs args{argc, argv};
  std::vector<std::vector<double>> vectors;

  vectors = read_points(args.input_file, args.num_dims);

  return 0;
}

std::vector<std::vector<double>> read_points(const std::string& filename, int dim) {
  std::ifstream in{filename};
  int num_points;
  std::string line;
  std::stringstream ss;
  int idx;

  std::getline(in, line);
  ss = std::stringstream(line);
  ss >> num_points;

  std::vector<std::vector<double>> vectors;
  vectors.reserve(num_points);

  while (!in.eof()) {
    std::vector<double> vect;
    double num;

    vect.reserve(dim);

    std::getline(in, line);
    if (line == "")
      continue;

    ss = std::stringstream(line);
    ss >> idx;

    for (int i = 0; i < dim; i++) {
      ss >> num;
      vect.push_back(num);
    }

    vectors.push_back(vect);
  }

  return vectors;
}
