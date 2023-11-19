#pragma once

#include "particle.h"

class SpatialPartitionTree2D {
  public:
    friend class Node;
    SpatialPartitionTree2D(double size);
    ~SpatialPartitionTree2D();

    void put(const Particle& particle);
    void traverse();
    void compute_centers();

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
