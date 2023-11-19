#pragma once

#include <ostream>

class Vector2D {
  public:
    Vector2D(double x = 0, double y = 0);

    double x() const;
    double y() const;
    double norm() const;

    Vector2D operator+(const Vector2D& rhs) const;
    Vector2D& operator+=(const Vector2D& rhs);

  private:
    double x_;
    double y_;
};

std::ostream& operator<<(std::ostream& out, const Vector2D& vector);
