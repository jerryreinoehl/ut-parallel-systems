#include "spatialpartitiontree.h"

#include <stack>

SpatialPartitionTree2D::SpatialPartitionTree2D(double size) : size_(size) {
  root_ = new Node{0, 0, size_};
}

SpatialPartitionTree2D::~SpatialPartitionTree2D() {
  std::stack<Node*> nodes;
  Node* node;
  nodes.push(root_);

  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();

    if (node != nullptr) {
      nodes.push(node->nw_);
      nodes.push(node->ne_);
      nodes.push(node->sw_);
      nodes.push(node->se_);
    }

    delete node;
  }
}

void SpatialPartitionTree2D::put(const Particle& particle) {
  root_->put(particle);
}

void SpatialPartitionTree2D::traverse() {
  std::stack<Node*> nodes;
  Node *node;
  nodes.push(root_);

  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();

    if (node->qty_ == 1) {
      printf("Region: x=%f, y=%f, size=%f\n", node->x_, node->y_, node->size_);
      std::cout << node->com_ << "\n\n";
    } else if (node->qty_ != 0) {
      nodes.push(node->nw_);
      nodes.push(node->ne_);
      nodes.push(node->sw_);
      nodes.push(node->se_);
    }
  }
}

SpatialPartitionTree2D::Node::Node(double x, double y, double size)
  : x_{x}, y_{y}, size_{size}
{}

void SpatialPartitionTree2D::Node::put(const Particle& particle) {
  if (qty_ == 0) {
    com_ = particle;
    qty_++;
    return;
  } else if (qty_ == 1) {
    subdivide();
    get_subregion(com_)->put(com_);
  }

  qty_++;
  get_subregion(particle)->put(particle);
}

void SpatialPartitionTree2D::Node::subdivide() {
  double half_size = size_ / 2;
  nw_ = new Node{x_, y_, half_size};
  ne_ = new Node{x_ + half_size, y_, half_size};
  sw_ = new Node{x_, y_ + half_size, half_size};
  se_ = new Node{x_ + half_size, y_ + half_size, half_size};
}

SpatialPartitionTree2D::Node *SpatialPartitionTree2D::Node::get_subregion(const Particle& particle) const {
  double px = particle.get_x(), py = particle.get_y();
  double half_size = size_ / 2;

  if (px <= x_ + half_size && py <= y_ + half_size) {
    return nw_;
  } else if (px > x_ + half_size && py > y_ + half_size) {
    return se_;
  } else if (px <= x_ + half_size) {
    return sw_;
  } else {
    return ne_;
  }
}
