#include "mpi.h"

mpi::Ctx::Ctx(int *argc, char ***argv) {
  MPI_Init(argc, argv);
}

mpi::Ctx::~Ctx() {
  MPI_Finalize();
}

mpi::Ctx mpi::init(int *argc, char ***argv) {
  return Ctx{argc, argv};
}

int mpi::rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int mpi::size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

int mpi::barrier(MPI_Comm comm) {
  return MPI_Barrier(comm);
}
