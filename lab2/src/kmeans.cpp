#include "kmeans.h"

void
kmeans_sequential(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels
) {
  int num_clusters = args.num_clusters;
  int dim = args.num_dims;

  double dist;
  double min_dist;

  int cent;
  int min_cent;

  std::unique_ptr<int[]> counts{new int[num_clusters]};

  vect_clear(&counts[0], num_clusters);

  // Calculate closest centroid for each point.
  for (int pnt = 0; pnt < num_points; pnt++) {
    min_dist = vect_sq_dist(&points[pnt * dim], &centroids[0], dim);
    min_cent = 0;

    for (cent = 1; cent < num_clusters; cent++) {
      dist = vect_sq_dist(&points[pnt * dim], &centroids[cent * dim], dim);
      if (dist < min_dist) {
        min_dist = dist;
        min_cent = cent;
      }
    }

    labels[pnt] = min_cent;
    counts[min_cent]++;
  }

  // Clear each centroid.
  for (cent = 0; cent < num_clusters; cent++) {
    vect_clear(&centroids[cent * dim], dim);
  }

  // Calculate new centroids.
  for (int pnt = 0, cent; pnt < num_points; pnt++) {
    cent = labels[pnt];
    vect_add(&centroids[cent * dim], &points[pnt * dim], dim);
  }

  for (cent = 0; cent < num_clusters; cent++) {
    vect_div(&centroids[cent * dim], counts[cent], dim);
  }

  // Print centroids
  for (cent = 0; cent < num_clusters; cent++) {
    vect_print(&centroids[cent * dim], dim);
  }

  printf("=====================\n");
}

double vect_sq_dist(double *a, double *b, int dim) {
  double dist = 0;
  double diff = 0;

  for (int i = 0; i < dim; i++) {
    diff = a[i] - b[i];
    dist += diff * diff;
  }

  return dist;
}
