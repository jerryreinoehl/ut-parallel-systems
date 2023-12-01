#include <iostream>
#include <argparse.h>
#include <threads.h>
#include <io.h>
#include <chrono>
#include <cstring>
#include "operators.h"
#include "helpers.h"
#include "prefix_sum.h"

using namespace std;

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    bool sequential = false;
    if (opts.n_threads == 0) {
        opts.n_threads = 1;
        sequential = true;
    }

    // Setup args & read input data
    int n_vals;
    int *input_vals, *output_vals;
    read_file(&opts, &n_vals, &input_vals, &output_vals);

    // Setup threads
    pthread_t *threads = sequential ? NULL : alloc_threads(opts.n_threads);;
    prefix_sum_args_t *ps_args = alloc_args(opts.n_threads);

    // Dynamically determine which barrier implementation we want to use based
    // on the value of the `spin` option.
    void *barrier;
    if (opts.spin)
      barrier = new SpinBarrier{opts.n_threads};
    else
      barrier = new PThreadBarrier{opts.n_threads};

    //"op" is the operator you have to use, but you can use "add" to test
    int (*scan_operator)(int, int, int);
    scan_operator = op;
    //scan_operator = add;

    fill_args(ps_args, opts.n_threads, n_vals, input_vals, output_vals,
        opts.spin, scan_operator, opts.n_loops, barrier);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    if (sequential)  {
        //sequential prefix scan
        output_vals[0] = input_vals[0];
        for (int i = 1; i < n_vals; ++i) {
            //y_i = y_{i-1}  <op>  x_i
            output_vals[i] = scan_operator(output_vals[i-1], input_vals[i], ps_args->n_loops);
        }
    }
    else {
        if (opts.spin)
          start_threads(threads, opts.n_threads, ps_args, compute_prefix_sum<SpinBarrier>);
        else
          start_threads(threads, opts.n_threads, ps_args, compute_prefix_sum<PThreadBarrier>);

        // Wait for threads to finish
        join_threads(threads, opts.n_threads);
    }

    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "time: " << diff.count() << std::endl;

    // Write output data
    write_file(&opts, &(ps_args[0]));

    // Free other buffers
    free(threads);
    free(ps_args);

    if (opts.spin)
      delete static_cast<SpinBarrier*>(barrier);
    else
      delete static_cast<PThreadBarrier*>(barrier);
}
