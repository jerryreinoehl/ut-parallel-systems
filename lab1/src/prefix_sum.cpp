#include "prefix_sum.h"
#include "helpers.h"
#include <unistd.h>
#include <pthread.h>

void print_array(int *a, int count) {
  printf("[");
  for (int i = 0; i < count-1; i++) {
    printf("%d, ", a[i]);
  }
  printf("%d]\n", a[count-1]);
}
