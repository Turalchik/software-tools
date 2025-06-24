#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

constexpr double kDx = 0.01;

inline double evalFunc(double x, double y) {
    return x * (std::sin(x) + std::cos(y));
}

void initField(std::vector<double>& field, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        double xi = i * kDx;
        for (int j = 0; j < cols; ++j) {
            field[i * cols + j] = evalFunc(xi, j * kDx);
        }
    }
}

void computeDx(const std::vector<double>& in, std::vector<double>& out,
               int startRow, int rowCount, int cols)
{
    for (int i = startRow; i < startRow + rowCount; ++i) {
        int rowOffset = i * cols;
        for (int j = 0; j < cols; ++j) {
            double left  = (j > 0)        ? in[rowOffset + j - 1] : in[rowOffset + j];
            double right = (j < cols - 1) ? in[rowOffset + j + 1] : in[rowOffset + j];
            out[rowOffset + j] = (right - left) / ( (j==0 || j==cols-1) ? kDx : (2 * kDx) );
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::vector<int> gridSizes = {10, 100, 1000};

    for (int N : gridSizes) {
        int rows = N, cols = N;
        int baseRows = rows / worldSize;
        int extra   = rows % worldSize;
        int myRows  = baseRows + (worldRank < extra ? 1 : 0);
        int myStart = worldRank * baseRows + std::min(worldRank, extra);

        std::vector<double> fieldA(rows * cols);
        std::vector<double> fieldB(rows * cols);

        if (worldRank == 0) {
            initField(fieldA, rows, cols);

            auto t0 = std::chrono::high_resolution_clock::now();

            for (int pid = 1; pid < worldSize; ++pid) {
                int rowsToSend = baseRows + (pid < extra ? 1 : 0);
                int startRow   = pid * baseRows + std::min(pid, extra);
                MPI_Send(&rowsToSend, 1, MPI_INT, pid, 0, MPI_COMM_WORLD);
                MPI_Send(&startRow,   1, MPI_INT, pid, 0, MPI_COMM_WORLD);
                MPI_Send(fieldA.data() + startRow * cols,
                         rowsToSend * cols, MPI_DOUBLE, pid, 0, MPI_COMM_WORLD);
            }

            computeDx(fieldA, fieldB, 0, baseRows + (0 < extra ? 1 : 0), cols);

            for (int pid = 1; pid < worldSize; ++pid) {
                int rowsGot, startGot;
                MPI_Recv(&rowsGot,  1, MPI_INT, pid, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&startGot, 1, MPI_INT, pid, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(fieldB.data() + startGot * cols,
                         rowsGot * cols, MPI_DOUBLE, pid, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dt = t1 - t0;
            std::cout << "Grid " << rows << "×" << cols
                      << " -> time: " << dt.count() << " s\n";
        }
        else {
            MPI_Recv(&myRows,  1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&myStart, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(fieldA.data() + myStart * cols,
                     myRows * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            computeDx(fieldA, fieldB, myStart, myRows, cols);

            MPI_Send(&myRows,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&myStart, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(fieldB.data() + myStart * cols,
                     myRows * cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
