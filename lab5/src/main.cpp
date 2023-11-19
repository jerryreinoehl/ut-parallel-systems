#include "main.h"
#include "args.h"
#include "spatialpartitiontree.h"
#include "vector2d.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

int main(int argc, char **argv) {
  Args args{argc, argv};

  const double size = 4;
  double gravity = args.gravity();
  double threshold = args.threshold();
  double timestep = args.timestep();

  SpatialPartitionTree2D spt{size};
  std::vector<Particle> particles = read_particles(args.input());
  Vector2D force;

  clock_t start = clock(), end;

  for (int step = 0; step < args.steps(); step++) {
    spt.reset();
    spt.put(particles);
    spt.compute_centers();

    for (auto& particle : particles) {
      if (!spt.in_bounds(particle)) {
        particle.set_mass(-1);
        continue;
      }

      force = spt.compute_force(particle, threshold, gravity);
      particle.apply_force(force, timestep);
    }
  }

  end = clock();
  std::cout << std::setprecision(6) << ((double)end - start) / CLOCKS_PER_SEC << '\n';

  write_particles(args.output(), particles);

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
