#include "main.h"
#include "args.h"
#include "spatialpartitiontree.h"
#include "vector2d.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

#include <chrono>

int main(int argc, char **argv) {
  Args args{argc, argv};

  if (args.sequential()) {
    seq_barnes_hut(args);
  } else {
    auto ctx = mpi::init(&argc, &argv);
    mpi_barnes_hut(args);
  }

  return 0;
}

void seq_barnes_hut(const Args& args) {
  const double size = 4;
  double gravity = args.gravity();
  double threshold = args.threshold();
  double timestep = args.timestep();
  double rlimit = args.rlimit();
  GLFWwindow *window{};

  if (args.visual()) {
    window = init_window();
  }

  SpatialPartitionTree2D spt{size};
  std::vector<Particle> particles = read_particles(args.input());
  Vector2D force;

  auto start = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < args.steps(); step++) {
    if (args.visual()) {
      draw(window, particles, spt);
    }

    spt.reset();
    spt.put(particles);
    spt.compute_centers();

    for (auto& particle : particles) {
      if (!spt.in_bounds(particle)) {
        particle.set_mass(-1);
        continue;
      }

      force = spt.compute_force(particle, threshold, gravity, rlimit);
      particle.apply_force(force, timestep);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << std::setprecision(8) << (double) duration / 1'000'000'000 << '\n';

  write_particles(args.output(), particles);
}

void mpi_barnes_hut(const Args& args) {
  int rank = mpi::rank();
  int num_procs = mpi::size();
  mpi::MessageGroup mg;
  GLFWwindow *window{};

  const double size = 4;
  double gravity = args.gravity();
  double threshold = args.threshold();
  double timestep = args.timestep();
  double rlimit = args.rlimit();

  SpatialPartitionTree2D spt{size};
  std::vector<Particle> particles;
  Particle *particle;
  int num_particles;
  int part_start, part_end;
  Vector2D force;

  bool visualize = rank == 0 && args.visual();

  if (visualize) {
    window = init_window();
  }

  // First process reads particles and broadcasts to others.
  if (rank == 0) {
    particles = read_particles(args.input());
    num_particles = particles.size();
  }

  mpi::broadcast(&num_particles, 1, 0);

  // Other processes updates their particles.
  if (rank != 0) {
    particles.resize(num_particles);
  }

  mpi::broadcast(&particles[0], num_particles, 0);

  part_start = num_particles * rank / num_procs;
  part_end = num_particles * (rank + 1) / num_procs;

  auto start = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < args.steps(); step++) {
    if (visualize) {
      draw(window, particles, spt);
    }

    spt.reset();
    spt.put(particles);
    spt.compute_centers();

    for (int i = part_start; i < part_end; i++) {
      particle = &particles[i];
      if (!spt.in_bounds(*particle)) {
        particle->set_mass(-1);
        continue;
      }

      force = spt.compute_force(*particle, threshold, gravity, rlimit);
      particle->apply_force(force, timestep);
    }

    for (int i = 0; i < num_procs; i++) {
      int pstart = num_particles * i / num_procs;
      int pend = num_particles * (i + 1) / num_procs;
      mg.broadcast(&particles[pstart], pend - pstart, i);
    }

    mg.wait();
  }

  if (rank == 0) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << std::setprecision(8) << (double) duration / 1'000'000'000 << '\n';
    write_particles(args.output(), particles);
  }
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

GLFWwindow *init_window() {
  const int width = 800, height = 800;
  GLFWwindow *window;

  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    return nullptr;
  }

  // Open a window and create is OpenGL context.
  window = glfwCreateWindow(width, height, "Simulation", NULL, NULL);
  if (window == NULL) {
    fprintf(stderr, "Failed to open GLFW window.\n");
    glfwTerminate();
    return nullptr;
  }

  glfwMakeContextCurrent(window); // Initialize GLEW
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return nullptr;
  }

  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
  return window;
}

void draw(GLFWwindow *window, const std::vector<Particle>& particles, const SpatialPartitionTree2D& spt) {
  glClear(GL_COLOR_BUFFER_BIT);
  drawOctreeBounds2D(spt);

  for (const auto& particle : particles) {
    drawParticle2D(particle);
  }

  // Swap buffers
  glfwSwapBuffers(window);
  glfwPollEvents();
}

void drawOctreeBounds2D(const SpatialPartitionTree2D& spt) {
  double x1, y1, x2, y2;

  for (const auto& bounds : spt.bounds()) {
    glBegin(GL_LINES);

    // Set the color of lines to be white.
    glColor3f(1.0f, 1.0f, 1.0f);

    x1 = bounds.x();
    y1 = bounds.y();
    x2 = x1 + bounds.z();
    y2 = y1 + bounds.z();

    // Transform bounds to fit on grid (-1, 1).
    x1 = x1 / 2 - 1;
    x2 = x2 / 2 - 1;
    y1 = y1 / 2 - 1;
    y2 = y2 / 2 - 1;

    glVertex2f(x1, y1);
    glVertex2f(x1, y2);

    glVertex2f(x1, y2);
    glVertex2f(x2, y2);

    glVertex2f(x2, y2);
    glVertex2f(x2, y1);

    glVertex2f(x2, y1);
    glVertex2f(x1, y1);

    glEnd();
  }
}

void drawParticle2D(const Particle& particle) {
  float x, y;
  float radius = 0.002 * particle.mass();
  if (radius < 0.005) {
    radius = 0.005;
  }

  x = particle.x() / 2 - 1;
  y = particle.y() / 2 - 1;

  glBegin(GL_TRIANGLE_FAN);
  glColor3f(0.0f, 0.67f, 1.0f);
  glVertex2f(x, y);

  for (int k = 0; k < 20; k++) {
    float angle = (float) k / 19 * 2 * 3.141592;
    glVertex2f(x + radius * cos(angle), y + radius * sin(angle));
  }
  glEnd();
}
