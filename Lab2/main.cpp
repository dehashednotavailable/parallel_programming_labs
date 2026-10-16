#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

struct matrix
{
    std::size_t size = 0;
    std::vector<double> values;
};

matrix read_matrix(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Не удалось открыть файл: " + filename);

    std::string size_text;
    if (!(file >> size_text) || size_text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("Размер матрицы должен быть целым положительным числом: " + filename);
    }

    const auto size_value = std::stoull(size_text);
    if (size_value == 0 || size_value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Неверный размер матрицы: " + filename);
    }

    matrix result;
    result.size = static_cast<std::size_t>(size_value);
    const std::size_t n = result.size;
    if (n > std::numeric_limits<std::size_t>::max() / n / (3 * sizeof(double))) {
        throw std::runtime_error("Слишком большой размер матрицы: " + filename);
    }

    for (std::size_t i = 0; i < n * n; ++i) {
        double value = 0.0;
        if (!(file >> value) || !std::isfinite(value)) {
            throw std::runtime_error("Недостаточно элементов или неверное число: " + filename);
        }
        result.values.push_back(value);
    }

    std::string extra;
    if (file >> extra) throw std::runtime_error("Лишние данные в матрице: " + filename);
    if (file.bad()) throw std::runtime_error("Ошибка чтения файла: " + filename);

    return result;
}

void multiply_matrix(const matrix& a, const matrix& b, matrix& result, int thread_count, int& threads_used)
{
    const std::size_t n = a.size;
#pragma omp parallel num_threads(thread_count)
    {
#pragma omp single
        threads_used = omp_get_num_threads();

#pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                const double value = a.values[static_cast<std::size_t>(i) * n + k];
                for (std::size_t j = 0; j < n; ++j) {
                    result.values[static_cast<std::size_t>(i) * n + j] += value * b.values[k * n + j];
                }
            }
        }
    }
}

void write_matrix(const matrix& result, const std::string& filename)
{
    for (const double value : result.values) {
        if (!std::isfinite(value)) throw std::runtime_error("Переполнение при вычислении результата.");
    }

    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Не удалось создать файл: " + filename);

    file << result.size << '\n' << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t i = 0; i < result.size; ++i) {
        for (std::size_t j = 0; j < result.size; ++j) {
            if (j != 0) file << ' ';
            file << result.values[i * result.size + j];
        }
        file << '\n';
    }
    file.close();
    if (!file) throw std::runtime_error("Ошибка записи файла: " + filename);
}

int main(int argc, char* argv[])
{
    if (argc != 4 && argc != 5) {
        std::cerr << "Использование: " << argv[0] << " A.txt B.txt C.txt [число_потоков]\n";
        return 1;
    }

    try {
        int thread_count = omp_get_max_threads();
        if (argc == 5) {
            std::size_t parsed = 0;
            thread_count = std::stoi(argv[4], &parsed);
            if (parsed != std::string(argv[4]).size()) throw std::runtime_error("Неверное число потоков.");
        }
        if (thread_count <= 0) throw std::runtime_error("Число потоков должно быть положительным.");

        const matrix a = read_matrix(argv[1]);
        const matrix b = read_matrix(argv[2]);
        if (a.size != b.size) throw std::runtime_error("Размеры матриц не совпадают.");

        matrix result;
        result.size = a.size;
        result.values.assign(a.size * a.size, 0.0);
        int threads_used = 1;

        const auto start = std::chrono::steady_clock::now();
        multiply_matrix(a, b, result, thread_count, threads_used);
        const auto finish = std::chrono::steady_clock::now();
        const double time_seconds = std::chrono::duration<double>(finish - start).count();

        write_matrix(result, argv[3]);

        const std::size_t matrix_bytes = 3 * a.size * a.size * sizeof(double);
        const long double scalar_operations = 2.0L * a.size * a.size * a.size;
        std::cout << "Результат записан в " << argv[3] << '\n';
        std::cout << "size=" << a.size << '\n';
        std::cout << "time_seconds=" << std::fixed << std::setprecision(9) << time_seconds << '\n';
        std::cout << "matrix_bytes=" << matrix_bytes << '\n';
        std::cout << "scalar_operations=" << std::setprecision(0) << scalar_operations << '\n';
        std::cout << "threads_requested=" << thread_count << '\n';
        std::cout << "threads_used=" << threads_used << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
