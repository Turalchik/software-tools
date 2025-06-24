#include <omp.h>
#include <iostream>
#include <vector>
#include <random>

static std::vector<std::vector<int>> make_matrix(int R, int C) {
    std::mt19937 eng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(1, 9);
    std::vector<std::vector<int>> m(R, std::vector<int>(C));
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            m[i][j] = dist(eng);
    return m;
}

static std::vector<std::vector<int>> matmul(const std::vector<std::vector<int>>& A,
                                            const std::vector<std::vector<int>>& B) {
    int RA = A.size(), CA = A[0].size();
    int CB = B[0].size();
    std::vector<std::vector<int>> C(RA, std::vector<int>(CB, 0));
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < RA; ++i) {
        for (int j = 0; j < CB; ++j) {
            int sum = 0;
            for (int k = 0; k < CA; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
    return C;
}

int main() {
    std::vector<std::pair<int,int>> dims{{10,10},{100,100},{1000,1000},{2000,2000}};
    for (auto [R, C] : dims) {
        auto M1 = make_matrix(R, C);
        auto M2 = make_matrix(C, R);
        double t0 = omp_get_wtime();
        auto M3 = matmul(M1, M2);
        double t1 = omp_get_wtime();
        std::cout << "Multiply " << R << "x" << C << " by " << C << "x" << R
                  << " took " << (t1 - t0) << " s\n";
    }
    return 0;
}
