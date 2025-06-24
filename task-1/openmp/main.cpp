#include <iostream>
#include <omp.h>

int main() {
    constexpr int kThreadCount = 4;
    omp_set_num_threads(kThreadCount);

#pragma omp parallel default(none) shared(std::cout)
    {
        const int thread_id = omp_get_thread_num();
        std::cout << "[Thread " << thread_id << "] Reporting in.\n";

#pragma omp barrier

        if (thread_id == 0) {
            std::cout << "[Master] All threads have arrived at the barrier.\n";
        }
    }

    return 0;
}
