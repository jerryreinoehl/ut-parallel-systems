#pragma once

#include <stdlib.h>
#include <pthread.h>
#include <spin_barrier.h>
#include <iostream>
#include "helpers.h"
#include <unistd.h>

// Computes the prefix sum of the values stored in `input_vals` into
// `output_vals`.
//
// This function assumes there are at least ceil(n_vals / n_threads) * n_threads
// elements allocated for both `input_vals` and `output_vals`, such
// that each block per thread has the same number of elements.
//
// While this function will run with any number of threads > 0, for optimal
// performance only max( floor(n_threads / 2), num-of-processor-threads) should
// be used.
template <typename Barrier>
void* compute_prefix_sum(void *a) {
  prefix_sum_args_t *args = (prefix_sum_args_t *)a;
  int *output = args->output_vals;
  int *input = args->input_vals;
  int t_id = args->t_id;
  int (*op)(int, int, int) = args->op;
  int n_vals = args->n_vals;
  int n_loops = args->n_loops;
  int n_threads = args->n_threads;

  Barrier *barrier = static_cast<Barrier*>(args->barrier);

  int stride;
  int block_size = (n_vals + n_threads - 1) / n_threads;  // = ceil(N/T)
  int block_start = t_id * block_size;
  int block_end = block_start + block_size - 1;

  // Perform prefix scan on our block of data.
  int x, y;
  x = input[block_start];
  output[block_start] = x;
  for (int i = 1; i < block_size; i++) {
    y = input[block_start + i];
    x = op(x, y, n_loops);
    output[block_start + i] = x;
  }

  barrier->wait();

  // Compute up-sweep.
  for (stride = 1; stride < n_threads; stride <<= 1) {
    if ((t_id + 1) % (stride << 1) == 0) {
      x = output[block_end - stride * block_size];
      y = output[block_end];
      output[block_end] = op(x, y, n_loops);
    }
    barrier->wait();
  }

  // Compute down-sweep.
  for (stride = stride >> 1; stride > 0; stride >>= 1) {
    if ((t_id + 1) % (stride << 1) == 0 && (t_id + stride) < n_threads) {
      x = output[block_end + stride * block_size];
      y = output[block_end];
      output[block_end + stride * block_size] = op(x, y, n_loops);
    }
    barrier->wait();
  }

  // Finalize prefix sum for our block. Add the sum from the previous block
  // to all of our elements except for the last.
  if (t_id > 0) {
    for (int i = 1; i < block_size; i++) {
      x = output[block_end - i];
      y = output[block_end - block_size];
      output[block_end - i] = op(x, y, n_loops);
    }
  }

  return 0;
}
