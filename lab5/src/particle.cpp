#include "particle.h"

#include <cmath>

Particle::Particle() {}

Particle::Particle(int id, double x, double y, double mass, double dx, double dy)
  : id_(id), x_(x), y_(y), mass_(mass), dx_(dx), dy_(dy)
{ }

int Particle::id() const {
  return id_;
}

double Particle::x() const {
  return x_;
}

double Particle::y() const {
  return y_;
}

double Particle::mass() const {
  return mass_;
}

double Particle::dx() const {
  return dx_;
}

double Particle::dy() const {
  return dy_;
}

void Particle::set_x(double x) {
  x_ = x;
}

void Particle::set_y(double y) {
  y_ = y;
}

void Particle::set_mass(double mass) {
  mass_ = mass;
}

void Particle::set_dx(double dx) {
  dx_ = dx;
}

void Particle::set_dy(double dy) {
  dy_ = dy;
}

double Particle::distance_to(const Particle& particle) const {
  double diff_x = particle.x_ - x_;
  double diff_y = particle.y_ - y_;
  return std::sqrt(diff_x * diff_x + diff_y * diff_y);
}

Vector2D Particle::force(const Particle& particle, double gravity) const {
  double dx, dy; // delta-x, -y
  double dist;
  double force_x, force_y;
  double gmm_d3; // G * M0 * M1 / d^3

  dx = particle.x_ - x_;
  dy = particle.y_ - y_;
  dist = std::sqrt(dx * dx + dy * dy);

  if (dist == 0) {
    return {0, 0};
  }

  gmm_d3 = gravity * mass_ * particle.mass_ / (dist * dist * dist);
  force_x = gmm_d3 * dx;
  force_y = gmm_d3 * dy;

  return {force_x, force_y};
}

void Particle::apply_force(const Vector2D& force, double dt) {
  double acc_x, acc_y; // Acceleration in each dimension.

  acc_x = force.x() / mass_;
  acc_y = force.y() / mass_;

  x_ = x_ + dx_ * dt + 0.5 * acc_x * dt * dt;
  y_ = y_ + dy_ * dt + 0.5 * acc_y * dt * dt;

  dx_ = dx_ + acc_x * dt;
  dy_ = dy_ + acc_y * dt;
}

std::string Particle::to_string() const {
  char buf[128];
  snprintf(buf, 128, "Particle(id=%d, x=%f, y=%f, mass=%f, dx=%f, dy=%f)", id_, x_, y_, mass_, dx_, dy_);
  return {buf};
}

std::ostream& operator<<(std::ostream& out, const Particle& particle) {
  return out << particle.to_string();
}
