#include "spatialpartitiontree.h"

#include <stack>
#include <queue>

SpatialPartitionTree2D::SpatialPartitionTree2D(double size) : size_(size) {
  root_ = new Node{0, 0, size_};
}

SpatialPartitionTree2D::SpatialPartitionTree2D(double size, const std::vector<Particle>& particles)
  : SpatialPartitionTree2D(size)
{
  put(particles);
}

SpatialPartitionTree2D::~SpatialPartitionTree2D() {
  std::stack<Node*> nodes;
  Node* node;
  nodes.push(root_);

  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();

    if (node == nullptr) {
      continue;
    }

    nodes.push(node->nw_);
    nodes.push(node->ne_);
    nodes.push(node->sw_);
    nodes.push(node->se_);

    delete node;
  }

  root_ = nullptr;
}

void SpatialPartitionTree2D::put(const Particle& particle) {
  if (!in_bounds(particle)) {
    return;
  }

  Node *node = root_;
  Node *subregion;

  while (true) {
    if (node->qty_ == 0) {
      node->com_ = particle;
      node->qty_ = 1;
      return;
    }

    if (node->qty_ == 1) {
      node->subdivide();
      subregion = node->get_subregion(node->com_);
      subregion->qty_ = 1;
      subregion->com_ = node->com_;
    }

    node->qty_++;
    subregion = node->get_subregion(particle);
    node = subregion;
  }
}

void SpatialPartitionTree2D::put(const std::vector<Particle>& particles) {
  for (const auto& particle: particles) {
    put(particle);
  }
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

  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();

    if (node->qty_ <= 1) {
      continue;
    }

    node->compute_center();
  }
}

Vector2D SpatialPartitionTree2D::compute_force(
  const Particle& particle, double threshold, double gravity, double rlimit
) const {
  std::stack<Node*> nodes;
  Node *node;

  double dist;
  double dx, dy;  // delta-x, -y
  double gmm_d3;  // G * M0 * M1 / d^3
  double force_x = 0, force_y = 0;

  nodes.push(root_);

  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();

    if (node->qty_ == 0) {
      continue;
    }

    dist = particle.distance_to(node->com_);

    if (dist == 0) {
      continue;
    }

    if (node->qty_ == 1 || node->size_ / dist < threshold) {
      if (dist < rlimit) {
        dist = rlimit;
      }

      dx = node->com_.x() - particle.x();
      dy = node->com_.y() - particle.y();
      gmm_d3 = gravity * particle.mass() * node->com_.mass() / (dist * dist * dist);
      force_x += gmm_d3 * dx;
      force_y += gmm_d3 * dy;
    } else {
      nodes.push(node->nw_);
      nodes.push(node->ne_);
      nodes.push(node->sw_);
      nodes.push(node->se_);
    }
  }

  return {force_x, force_y};
}

bool SpatialPartitionTree2D::in_bounds(const Particle& particle) const {
  double px = particle.x(), py = particle.y();
  return px >= 0 && px <= size_ && py >= 0 && py <= size_;
}

void SpatialPartitionTree2D::reset() {
  std::stack<Node*> nodes;
  Node *node;

  nodes.push(root_);

  while (!nodes.empty()) {
    node = nodes.top();
    nodes.pop();

    if (node == nullptr) {
      continue;
    }

    node->qty_ = 0;

    nodes.push(node->nw_);
    nodes.push(node->ne_);
    nodes.push(node->sw_);
    nodes.push(node->se_);
  }
}

SpatialPartitionTree2D::Node::Node(double x, double y, double size)
  : x_{x}, y_{y}, size_{size}
{}

void SpatialPartitionTree2D::Node::compute_center() {
  double nw_x, ne_x, sw_x, se_x;
  double nw_y, ne_y, sw_y, se_y;
  double nw_mass, ne_mass, sw_mass, se_mass;
  double mass;

  nw_mass = nw_->com_.mass();
  nw_x = nw_mass * nw_->com_.x();
  nw_y = nw_mass * nw_->com_.y();

  ne_mass = ne_->com_.mass();
  ne_x = ne_mass * ne_->com_.x();
  ne_y = ne_mass * ne_->com_.y();

  sw_mass = sw_->com_.mass();
  sw_x = sw_mass * sw_->com_.x();
  sw_y = sw_mass * sw_->com_.y();

  se_mass = se_->com_.mass();
  se_x = se_mass * se_->com_.x();
  se_y = se_mass * se_->com_.y();

  mass = nw_mass + ne_mass + sw_mass + se_mass;

  com_.set_x((nw_x + ne_x + sw_x + se_x) / mass);
  com_.set_y((nw_y + ne_y + sw_y + se_y) / mass);
  com_.set_mass(mass);
}

std::string SpatialPartitionTree2D::Node::to_string() const {
  const int size{256};
  char buf[size];
  snprintf(buf, size, "Node(x=%f, y=%f, size=%f, com=%s, qty=%d)", x_, y_, size_, com_.to_string().c_str(), qty_);
  return {buf};
}

void SpatialPartitionTree2D::Node::subdivide() {
  double half_size = size_ / 2;

  // Subregions are either all allocated or all nullptr.
  if (nw_ == nullptr) {
    nw_ = new Node{x_, y_, half_size};
    ne_ = new Node{x_ + half_size, y_, half_size};
    sw_ = new Node{x_, y_ + half_size, half_size};
    se_ = new Node{x_ + half_size, y_ + half_size, half_size};
  }
}

SpatialPartitionTree2D::Node *SpatialPartitionTree2D::Node::get_subregion(const Particle& particle) const {
  double px = particle.x(), py = particle.y();
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
