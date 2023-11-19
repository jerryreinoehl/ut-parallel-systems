#include "spatialpartitiontree.h"

#include <stack>
#include <queue>

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

void SpatialPartitionTree2D::compute_centers() {
  std::queue<Node*> children;
  std::stack<Node*> nodes;
  Node *node;

  children.push(root_);

  // Place nodes on stack, deepest nodes on top.
  while (!children.empty()) {
    node = children.front();
    children.pop();

    nodes.push(node);
    if (node->qty_ > 1) {
      children.push(node->nw_);
      children.push(node->ne_);
      children.push(node->sw_);
      children.push(node->se_);
    }
  }

  double nw_x, ne_x, sw_x, se_x;
  double nw_y, ne_y, sw_y, se_y;
  double nw_mass, ne_mass, sw_mass, se_mass;
  double mass;

  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();

    if (node->qty_ <= 1) {
      continue;
    }

    nw_mass = node->nw_->com_.get_mass();
    nw_x = nw_mass * node->nw_->com_.get_x();
    nw_y = nw_mass * node->nw_->com_.get_y();

    ne_mass = node->ne_->com_.get_mass();
    ne_x = ne_mass * node->ne_->com_.get_x();
    ne_y = ne_mass * node->ne_->com_.get_y();

    sw_mass = node->sw_->com_.get_mass();
    sw_x = sw_mass * node->sw_->com_.get_x();
    sw_y = sw_mass * node->sw_->com_.get_y();

    se_mass = node->se_->com_.get_mass();
    se_x = se_mass * node->se_->com_.get_x();
    se_y = se_mass * node->se_->com_.get_y();

    mass = nw_mass + ne_mass + sw_mass + se_mass;

    node->com_.set_x((nw_x + ne_x + sw_x + se_x) / mass);
    node->com_.set_y((nw_y + ne_y + sw_y + se_y) / mass);
    node->com_.set_mass(mass);
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

std::string SpatialPartitionTree2D::Node::to_string() const {
  const int size{256};
  char buf[size];
  snprintf(buf, size, "Node(x=%f, y=%f, size=%f, com=%s, qty=%d)", x_, y_, size_, com_.to_string().c_str(), qty_);
  return {buf};
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
