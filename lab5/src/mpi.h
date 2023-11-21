#pragma once

#include <mpi.h>

#include <vector>

namespace mpi {
  class Ctx;
  class MessageGroup;

  Ctx init(int *argc, char ***argv);
  int rank(MPI_Comm comm = MPI_COMM_WORLD);
  int size(MPI_Comm comm = MPI_COMM_WORLD);
  int barrier(MPI_Comm comm = MPI_COMM_WORLD);
}

class mpi::Ctx {
  public:
    Ctx(int *argc, char ***argv);
    ~Ctx();
};

class mpi::MessageGroup {
  public:
    MessageGroup(MPI_Comm = MPI_COMM_WORLD);

    template <typename T>
    int broadcast(T *buffer, int count, int root) {
      MPI_Request request;
      int rc;

      rc = MPI_Ibcast(buffer, sizeof(T) * count, MPI_BYTE, root, comm_, &request);
      requests_.push_back(request);

      return rc;
    }

    int wait();

  private:
    MPI_Comm comm_;
    std::vector<MPI_Request> requests_{};
    std::vector<MPI_Status> statuses_{};
};
