#pragma once

#include "particle.h"

#include <string>
#include <vector>

std::vector<Particle> read_particles(const std::string& filename);
void write_particles(const std::string& filename, const std::vector<Particle>& particles);
