#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>


void populate_random(std::vector<int>& data, int max_value, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, max_value - 1);
    for (auto& elem : data) {
        elem = dist(rng);
    }
}

int sum_array(const int* arr, int length) {
    int sum = 0;
    for (int i = 0; i < length; ++i) {
        sum += arr[i];
    }
    return sum;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const std::vector<int> kTests = {10, 1000, 10'000'000};
    const unsigned int kRandomSeed = 42;

    for (int total_elements : kTests) {

        int base_block = total_elements / world_size;
        int extras = total_elements % world_size;
        int local_count = base_block + (world_rank < extras ? 1 : 0);

        std::vector<int> buffer(local_count);

        if (world_rank == 0) {

            std::vector<int> full_data(total_elements);
            populate_random(full_data, 10, kRandomSeed);

            int offset = 0;
            for (int pid = 1; pid < world_size; ++pid) {
                int chunk_size = base_block + (pid < extras ? 1 : 0);
                MPI_Send(&chunk_size, 1, MPI_INT, pid, 0, MPI_COMM_WORLD);
                MPI_Send(full_data.data() + offset, chunk_size, MPI_INT, pid, 0, MPI_COMM_WORLD);
                offset += chunk_size;
            }

            std::copy_n(full_data.data(), local_count, buffer.begin());

            auto t_start = std::chrono::high_resolution_clock::now();

            int total_sum = sum_array(buffer.data(), local_count);
            for (int pid = 1; pid < world_size; ++pid) {
                int partial;
                MPI_Recv(&partial, 1, MPI_INT, pid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_sum += partial;
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = t_end - t_start;

            std::cout << "Elements: " << total_elements
                      << ", Sum: " << total_sum
                      << ", Duration: " << elapsed.count() << "s\n";
        } else {

            MPI_Recv(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            buffer.resize(local_count);
            MPI_Recv(buffer.data(), local_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int partial_sum = sum_array(buffer.data(), local_count);
            MPI_Send(&partial_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
