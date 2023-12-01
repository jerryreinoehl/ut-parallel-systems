#ifndef _SPIN_BARRIER_H
#define _SPIN_BARRIER_H

#include <pthread.h>
#include <cstdio>
#include <unistd.h>
#include <thread>

// A simple wrapper around pthread_barrier_t so we can use a common barrier
// interface.
class PThreadBarrier {
  public:
    inline PThreadBarrier(int count) {
      pthread_barrier_init(&barrier_, nullptr, count);
    }

    inline ~PThreadBarrier() {
      pthread_barrier_destroy(&barrier_);
    }

    inline void wait() {
      pthread_barrier_wait(&barrier_);
    }

  private:
    pthread_barrier_t barrier_;
};

class SpinBarrier {
  public:
    SpinBarrier(int count);

    ~SpinBarrier();

    inline void wait();

  private:
    int count_;
    volatile int *go_;
    int go_len_;
    pthread_mutex_t count_lock_;
};

static const int HARDWARE_CONCURRENCY = std::thread::hardware_concurrency();

inline void SpinBarrier::wait() {
  int count;
  int go;

  pthread_mutex_lock(&count_lock_);
  count = --count_;
  go = !go_[count];

  if (count == 0) {
    pthread_mutex_unlock(&count_lock_);
    count_ = go_len_;
    for (int i = go_len_ - 1; i >= 0; i--)
      go_[i] = go;
  } else {
    pthread_mutex_unlock(&count_lock_);
    while(go_[count] != go) {
      if (count >= HARDWARE_CONCURRENCY) {
        sleep(0); // yield cpu
      }
    }
  }
}

#endif
