#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

constexpr int MAX_N = 2000;

double computeValue(int i, int j) {
    return static_cast<double>(std::rand() % 10);
}

void multiplyChunk(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C,
                   int start, int count, int N) {
    for (int i = 0; i < count; ++i) {
        int row = start + i;
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + j];
            }
            C[row * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> dims = {10, 100, 1000, 2000};
    std::srand(42);

    for (int N : dims) {
        std::vector<double> A(N * N), B(N * N), C(N * N);

        if (rank == 0) {
            for (int i = 0; i < N * N; ++i) {
                A[i] = B[i] = computeValue(i, i);
                C[i] = 0.0;
            }

            int base = N / size;
            int rem  = N % size;
            int offset = base;

            for (int p = 1; p < size; ++p) {
                int rows = (p < size - 1) ? base : (base + rem);
                MPI_Send(&offset, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(&rows,   1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(A.data() + offset * N, rows * N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Send(B.data(), N * N,       MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                offset += rows;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            multiplyChunk(A, B, C, 0, base, N);

            for (int p = 1; p < size; ++p) {
                int start, count;
                MPI_Recv(&start, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&count, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(C.data() + start * N, count * N, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            auto t2 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t2 - t1).count();
            std::cout << "N=" << N << " Time=" << dt << "s\n";

        } else {
            int start, count;
            MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            A.resize(N * N);
            MPI_Recv(A.data() + start * N, count * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            B.resize(N * N);
            MPI_Recv(B.data(), N * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            multiplyChunk(A, B, C, start, count, N);

            MPI_Send(&start, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(C.data() + start * N, count * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
