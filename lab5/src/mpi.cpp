#include "mpi.h"

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

mpi::Ctx::Ctx(int *argc, char ***argv) {
  MPI_Init(argc, argv);
}

mpi::Ctx::~Ctx() {
  MPI_Finalize();
}

mpi::MessageGroup::MessageGroup(MPI_Comm comm) : comm_{comm} {}

int mpi::MessageGroup::wait() {
  int rc;

  statuses_.clear();
  statuses_.resize(requests_.size());
  rc = MPI_Waitall(requests_.size(), &requests_[0], &statuses_[0]);
  requests_.clear();

  return rc;
}
