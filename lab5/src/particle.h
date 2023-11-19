#pragma once

#include "vector2d.h"

#include <iostream>

class Particle {
  public:
    Particle();
    Particle(int id, double x, double y, double mass, double dx, double dy);

    int get_id() const;
    double get_x() const;
    double get_y() const;
    double get_mass() const;
    double get_dx() const;
    double get_dy() const;

    void set_x(double x);
    void set_y(double y);
    void set_mass(double mass);
    void set_dx(double dx);
    void set_dy(double dy);

    double distance_to(const Particle& particle) const;
    Vector2D force(const Particle& particle, double gravity) const;

    std::string to_string() const;

  private:
    int id_{0};
    double x_{0};
    double y_{0};
    double mass_{0};
    double dx_{0};
    double dy_{0};
};

std::ostream& operator<<(std::ostream& out, const Particle& particle);
