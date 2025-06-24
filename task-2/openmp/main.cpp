#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>


static std::vector<int> make_random_vector(std::size_t len, int maxVal, unsigned int seed) {
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int> dist(0, maxVal - 1);
    std::vector<int> vec(len);
    for (auto& v : vec) {
        v = dist(engine);
    }
    return vec;
}

int main() {
    constexpr unsigned int kSeed = 42;
    constexpr int kMaxValue = 10;
    const std::vector<std::size_t> kSizes = {10, 1'000, 10'000'000};

    for (auto n : kSizes) {
        auto data = make_random_vector(n, kMaxValue, kSeed);

        double t0 = omp_get_wtime();

        long long sum = 0;
#pragma omp parallel for reduction(+:sum) default(none) shared(data, n)
        for (std::size_t i = 0; i < n; ++i) {
            sum += data[i];
        }

        double t1 = omp_get_wtime();

        std::cout << "[N=" << n << "] Sum=" << sum
                  << "  Time=" << (t1 - t0) << "s\n";
    }

    return 0;
}
