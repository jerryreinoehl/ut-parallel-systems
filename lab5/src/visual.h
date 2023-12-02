#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include <GL/glut.h>

#include "spatialpartitiontree.h"

GLFWwindow *init_window();
void draw(GLFWwindow *window, const std::vector<Particle>& particles, const SpatialPartitionTree2D& spt);
void drawOctreeBounds2D(const SpatialPartitionTree2D& spt);
void drawParticle2D(const Particle& particle);
