#include <omp.h>
#include <cmath>
#include <iostream>
#include <vector>

static double evaluate(double x, double y) {
    return x * (std::sin(x) + std::cos(y));
}

void compute_dx(const std::vector<std::vector<double>>& in,
                std::vector<std::vector<double>>& out,
                double delta) {
    int rows = in.size();
    int cols = in[0].size();
#pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double left  = (c > 0)        ? in[r][c - 1] : in[r][c];
            double right = (c < cols - 1) ? in[r][c + 1] : in[r][c];
            out[r][c] = (right - left) / ((c == 0 || c == cols - 1) ? delta : (2 * delta));
        }
    }
}

int main() {
    std::vector<int> sizes = {10, 100, 1000};
    constexpr double dx = 0.01;

    for (auto N : sizes) {
        int R = N, C = N;
        std::vector<std::vector<double>> grid(R, std::vector<double>(C));
        std::vector<std::vector<double>> deriv(R, std::vector<double>(C));

#pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                grid[r][c] = evaluate(r * dx, c * dx);
            }
        }

        double t_start = omp_get_wtime();
        compute_dx(grid, deriv, dx);
        double t_end   = omp_get_wtime();

        std::cout << "Size " << R << "x" << C
                  << " -> Time: " << (t_end - t_start) << " s\n";
    }

    return 0;
}
