#pragma once

#include <iostream>

class Particle {
  public:
    Particle(int id, double x, double y, double mass, double dx, double dy);

    int get_id() const;
    double get_x() const;
    double get_y() const;
    double get_mass() const;
    double get_dx() const;
    double get_dy() const;

    std::string to_string() const;

  private:
    int id_;
    double x_;
    double y_;
    double mass_;
    double dx_;
    double dy_;
};

std::ostream& operator<<(std::ostream& out, const Particle& particle);
