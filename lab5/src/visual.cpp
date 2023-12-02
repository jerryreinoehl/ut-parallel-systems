#include "visual.h"

#include <stdio.h>

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
  } else if (radius > 0.1) {
    radius = 0.1;
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
