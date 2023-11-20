#pragma once

#include "args.h"
#include "particle.h"

#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include <GL/glut.h>

void seq_barnes_hut(const Args& args, GLFWwindow *window);
std::vector<Particle> read_particles(const std::string& filename);
void write_particles(const std::string& filename, const std::vector<Particle>& particles);

GLFWwindow *init_window();
void draw(GLFWwindow *window, const std::vector<Particle>& particles);
void drawOctreeBounds2D();
void drawParticle2D(const Particle& particle);
