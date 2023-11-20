#pragma once

#include "args.h"
#include "particle.h"

#include <string>
#include <vector>

void seq_barnes_hut(const Args& args);
std::vector<Particle> read_particles(const std::string& filename);
void write_particles(const std::string& filename, const std::vector<Particle>& particles);
