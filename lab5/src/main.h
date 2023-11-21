#pragma once

#include "args.h"
#include "particle.h"
#include "spatialpartitiontree.h"

#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include <GL/glut.h>

#include <mpi.h>
#include "mpi.h"

void seq_barnes_hut(const Args& args);
void mpi_barnes_hut(const Args& args);

std::vector<Particle> read_particles(const std::string& filename);
void write_particles(const std::string& filename, const std::vector<Particle>& particles);

GLFWwindow *init_window();
void draw(GLFWwindow *window, const std::vector<Particle>& particles, const SpatialPartitionTree2D& spt);
void drawOctreeBounds2D(const SpatialPartitionTree2D& spt);
void drawParticle2D(const Particle& particle);
