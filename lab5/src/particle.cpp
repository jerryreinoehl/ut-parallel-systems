#include "particle.h"

Particle::Particle() {}

Particle::Particle(int id, double x, double y, double mass, double dx, double dy)
  : id_(id), x_(x), y_(y), mass_(mass), dx_(dx), dy_(dy)
{ }

int Particle::get_id() const {
  return id_;
}

double Particle::get_x() const {
  return x_;
}

double Particle::get_y() const {
  return y_;
}

double Particle::get_mass() const {
  return mass_;
}

double Particle::get_dx() const {
  return dx_;
}

double Particle::get_dy() const {
  return dy_;
}

std::string Particle::to_string() const {
  char buf[128];
  snprintf(buf, 128, "Particle(id=%d, x=%f, y=%f, mass=%f, dx=%f, dy=%f)", id_, x_, y_, mass_, dx_, dy_);
  return {buf};
}

std::ostream& operator<<(std::ostream& out, const Particle& particle) {
  return out << particle.to_string();
}
