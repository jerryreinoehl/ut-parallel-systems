#pragma once

#include "particle.h"
#include "vector2d.h"

#include <vector>

class SpatialPartitionTree2D {
  public:
    friend class Node;
    SpatialPartitionTree2D(double size);
    ~SpatialPartitionTree2D();

    void put(const Particle& particle);
    void put(const std::vector<Particle>& particles);
    void traverse();
    void compute_centers();

    Vector2D compute_force(const Particle& particle, double threshold, double gravity) const;

  private:
    class Node;

    double size_;
    Node *root_;
};

class SpatialPartitionTree2D::Node {
  friend class SpatialPartitionTree2D;

  public:
    Node(double x, double y, double size);

    void put(const Particle& particle);
    void compute_center();
    std::string to_string() const;

  private:
    double x_;
    double y_;
    double size_;  // Length and width of this region.
    Particle com_; // Center of mass.
    int qty_{0};   // Number of bodies in this region.
    Node *nw_{nullptr};
    Node *ne_{nullptr};
    Node *sw_{nullptr};
    Node *se_{nullptr};

    void subdivide();
    Node *get_subregion(const Particle& particle) const;
};