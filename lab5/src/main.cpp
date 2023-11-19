#include "main.h"
#include "args.h"
#include "spatialpartitiontree.h"

#include <fstream>
#include <sstream>
#include <iomanip>

int main(int argc, char **argv) {
  Args args{argc, argv};

  std::vector<Particle> particles = read_particles(args.input());

  for (const auto& particle : particles) {
    std::cout << particle << '\n';
  }

  return 0;
}

std::vector<Particle> read_particles(const std::string& filename) {
  std::vector<Particle> particles;
  std::ifstream in{filename};
  std::stringstream ss;
  std::string line;

  std::getline(in, line);

  while (!in.eof()) {
    int id;
    double x, y, mass, dx, dy;

    std::getline(in, line);
    if (line == "")
      continue;

    ss = std::stringstream(line);
    ss >> id >> x >> y >> mass >> dx >> dy;

    particles.push_back({id, x, y, mass, dx, dy});
  }

  return particles;
}

void write_particles(const std::string& filename, const std::vector<Particle>& particles) {
  std::ofstream out{filename};

  out << std::scientific << std::setprecision(6);

  out << particles.size() << '\n';

  for (const auto& particle : particles) {
    out << particle.id() << '\t' << particle.x() << '\t' << particle.y() << '\t'
        << particle.mass() << '\t' << particle.dx() << '\t' << particle.dy() << '\n';
  }
}
