#pragma once

#include <mpi.h>

namespace mpi {
  class Ctx {
    public:
      Ctx(int *argc, char ***argv);
      ~Ctx();
  };

  Ctx init(int *argc, char ***argv);
  int rank(MPI_Comm comm = MPI_COMM_WORLD);
  int size(MPI_Comm comm = MPI_COMM_WORLD);
  int barrier(MPI_Comm comm = MPI_COMM_WORLD);
}
