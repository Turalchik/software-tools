#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

int main() {
    const int MAX_THREADS = omp_get_max_threads();
    std::vector<int> lengths(MAX_THREADS, 0);
    std::vector<std::string> messages(MAX_THREADS);
    omp_lock_t lock;
    omp_init_lock(&lock);

#pragma omp parallel
    {
        int rank = omp_get_thread_num();
        int size = omp_get_num_threads();

#pragma omp critical
        std::cout << "Поток " << rank << " из " << size << " готов\n";

        if (rank != 0) {
            std::string message = "Поток " + std::to_string(rank) + " приветствует главный!";
            omp_set_lock(&lock);
            lengths[rank] = message.size();
            messages[rank] = message;
            omp_unset_lock(&lock);
        }

#pragma omp barrier

        if (rank == 0) {
            int count = 0;
            for (int i = 1; i < size; i++) {
                if (lengths[i] > 0) {
                    std::cout << "Получено от " << i << ": " << messages[i] << "\n";
                    count++;
                }
            }
            std::cout << "Всего сообщений: " << count << "\n";
        }
    }

    omp_destroy_lock(&lock);
    return 0;
}