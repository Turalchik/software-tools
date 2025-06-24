#include <mpi.h>

int main(int argc, char *argv[])
{
	const int ITERATIONS = 3;
	int world_rank, world_size;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	for (int round = 0; round < ITERATIONS; ++round) {
		printf("Process %d out of %d is alive\n", world_rank, world_size);
	}

	MPI_Finalize();
	return 0;
}
