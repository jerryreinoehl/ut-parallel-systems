#pragma once

#include "particle.h"
#include "vector2d.h"
#include "vector3d.h"

#include <vector>

class SpatialPartitionTree2D {
  public:
    friend class Node;
    SpatialPartitionTree2D(double size);
    SpatialPartitionTree2D(double size, const std::vector<Particle>& particles);
    ~SpatialPartitionTree2D();

    void put(const Particle& particle);
    void put(const std::vector<Particle>& particles);
    void traverse();
    void compute_centers();

    Vector2D compute_force(const Particle& particle, double threshold, double gravity, double rlimit) const;

    bool in_bounds(const Particle& particle) const;

    // Resets all allocated nodes to empty without freeing them. When reusing the tree,
    // this is much faster than creating a new one and reallocating new nodes, however,
    // this may increase memory use.
    void reset();

    std::vector<Vector3D> bounds() const;

  private:
    class Node;

    double size_;
    Node *root_;
};

class SpatialPartitionTree2D::Node {
  friend class SpatialPartitionTree2D;

  public:
    Node(double x, double y, double size);

    void compute_center();
    std::string to_string() const;

  private:
    double x_;
    double y_;
    double size_;  // Length and width of this region.
    Particle com_; // Center of mass.
    int qty_{};    // Number of bodies in this region.
    Node *nw_{};
    Node *ne_{};
    Node *sw_{};
    Node *se_{};

    void subdivide();
    Node *get_subregion(const Particle& particle) const;
};
