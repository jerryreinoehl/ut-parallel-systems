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
  GLFWwindow *window{};

  if (args.visual()) {
    window = init_window();
  }

  seq_barnes_hut(args, window);

  return 0;
}

void seq_barnes_hut(const Args& args, GLFWwindow *window) {
  const double size = 4;
  double gravity = args.gravity();
  double threshold = args.threshold();
  double timestep = args.timestep();
  double rlimit = args.rlimit();

  SpatialPartitionTree2D spt{size};
  std::vector<Particle> particles = read_particles(args.input());
  Vector2D force;

  clock_t start = std::clock(), end;

  for (int step = 0; step < args.steps(); step++) {
    if (args.visual()) {
      draw(window, particles);
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

  end = std::clock();
  std::cout << std::setprecision(6) << ((double)end - start) / CLOCKS_PER_SEC << '\n';

  write_particles(args.output(), particles);
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
  const int width = 600, height = 600;
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

void draw(GLFWwindow *window, const std::vector<Particle>& particles) {
  glClear(GL_COLOR_BUFFER_BIT);
  drawOctreeBounds2D();

  for (const auto& particle : particles) {
    drawParticle2D(particle);
  }

  // Swap buffers
  glfwSwapBuffers(window);
  glfwPollEvents();
}

void drawOctreeBounds2D() {
  glBegin(GL_LINES);

  // Set the color of lines to be white.
  glColor3f(1.0f, 1.0f, 1.0f);

  // Specify the start point's coordinates.
  glVertex2f(-1, 0);

  // Specify the end point's coordinates.
  glVertex2f(1, 0);

  // Do the same for the verticle line.
  glVertex2f(0, -1);
  glVertex2f(0, 1);

  glEnd();
}

void drawParticle2D(const Particle& particle) {
  float x, y;
  float radius = 0.005 * particle.mass();
  //radius = 0.001;

  x = 2 * particle.x() / 4 - 1;
  y = 2 * particle.y() / 4 - 1;

  glBegin(GL_TRIANGLE_FAN);
  glColor3f(0.0f, 0.67f, 1.0f);
  glVertex2f(x, y);

  for (int k = 0; k < 20; k++) {
    float angle = (float) k / 19 * 2 * 3.141592;
    glVertex2f(x + radius * cos(angle), y + radius * sin(angle));
  }
  glEnd();
}
