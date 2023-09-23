#include "kmeans.h"
#include <cstdlib>
#include <string.h>
#include <cmath>

void
kmeans_sequential(
  const KmeansArgs& args,
  int num_points,
  std::unique_ptr<double[]>& centroids,
  std::unique_ptr<double[]>& points,
  std::unique_ptr<int[]>& labels,
  int *num_iters
) {
  int num_clusters = args.num_clusters;
  int dim = args.num_dims;
  int iters = 0;

  // Instead of sqrt'ing the distance every time we square the threshold once.
  double threshold = args.threshold * args.threshold;

  double dist;
  double min_dist;
  int cent;
  int min_cent;

  bool converged = false;

  std::unique_ptr<int[]> counts{new int[num_clusters]};
  std::unique_ptr<double[]> centroids_prev{new double[num_clusters * dim]};

  while (!converged && iters < args.max_iters) {
    iters++;

    // Put a copy of our centroids in `centroids_prev`. We'll use this for
    // convergence test.
    memcpy(centroids_prev.get(), centroids.get(), num_clusters * dim * sizeof(double));

    // Reset point counts for each centroid.
    vect_clear(&counts[0], num_clusters);

    // Calculate closest centroid for each point and number of points mapped
    // to each centroid.
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

    // Clear centroids.
    bzero(centroids.get(), num_clusters * dim * sizeof(double));

    // Calculate new centroids.
    for (int pnt = 0, cent; pnt < num_points; pnt++) {
      cent = labels[pnt];
      vect_add(&centroids[cent * dim], &points[pnt * dim], dim);
    }

    for (cent = 0; cent < num_clusters; cent++) {
      vect_div(&centroids[cent * dim], counts[cent], dim);
    }

    // We have converged if each of the new centroids are within a distance
    // `threshold` of their previous position.
    converged = true;
    for (cent = 0; cent < num_clusters; cent++) {
      dist = vect_sq_dist(&centroids[cent * dim], &centroids_prev[cent * dim], dim);
      if (dist > threshold)
        converged = false;
    }
  }

  *num_iters = iters;
}
