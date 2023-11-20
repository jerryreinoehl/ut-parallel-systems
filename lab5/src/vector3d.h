#pragma once

#include <ostream>

class Vector3D {
  public:
    Vector3D(double x = 0, double y = 0, double z = 0);

    double x() const;
    double y() const;
    double z() const;
    double norm() const;

    Vector3D operator+(const Vector3D& rhs) const;
    Vector3D& operator+=(const Vector3D& rhs);

  private:
    double x_;
    double y_;
    double z_;
};

std::ostream& operator<<(std::ostream& out, const Vector3D& vector);
