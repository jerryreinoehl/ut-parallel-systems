#include <spin_barrier.h>
#include <pthread.h>
#include <cstring>

SpinBarrier::SpinBarrier(int count) {
  count_ = count;
  go_ = new int[count]();
  go_len_ = count;
  pthread_mutex_init(&count_lock_, nullptr);
}

SpinBarrier::~SpinBarrier() {
  delete [] go_;
  pthread_mutex_destroy(&count_lock_);
}
