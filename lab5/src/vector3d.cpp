#include "vector3d.h"

#include <cmath>

Vector3D::Vector3D(double x, double y, double z) : x_{x}, y_{y}, z_{z} {}

double Vector3D::x() const {
  return x_;
}

double Vector3D::y() const {
  return y_;
}

double Vector3D::z() const {
  return z_;
}

double Vector3D::norm() const {
  return std::sqrt(x_ * x_ + y_ * y_ + z_ * z_);
}

Vector3D Vector3D::operator+(const Vector3D& rhs) const {
  return {x_ + rhs.x_, y_+ rhs.y_, z_ + rhs.z_};
}

Vector3D& Vector3D::operator+=(const Vector3D& rhs) {
  x_ += rhs.x_;
  y_ += rhs.y_;
  z_ += rhs.z_;
  return *this;
}

std::ostream& operator<<(std::ostream& out, const Vector3D& vector) {
  out << "Vector(x=" << vector.x() << ", y=" << vector.y() << ", z=" << vector.z() << ")";
  return out;
}
