#include "vector2d.h"

#include <cmath>

Vector2D::Vector2D(double x, double y) : x_{x}, y_{y} {}

double Vector2D::x() const {
  return x_;
}

double Vector2D::y() const {
  return y_;
}

double Vector2D::norm() const {
  return std::sqrt(x_ * x_ + y_ * y_);
}

Vector2D Vector2D::operator+(const Vector2D& rhs) const {
  return {x_ + rhs.x_, y_+ rhs.y_};
}

Vector2D& Vector2D::operator+=(const Vector2D& rhs) {
  x_ += rhs.x_;
  y_ += rhs.y_;
  return *this;
}

std::ostream& operator<<(std::ostream& out, const Vector2D& vector) {
  out << "Vector(x=" << vector.x() << ", y=" << vector.y() << ")";
  return out;
}
