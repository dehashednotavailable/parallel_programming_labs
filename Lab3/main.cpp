#include <mpi.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

void multiply_rows(const std::vector<double>& a, const std::vector<double>& b,
                   std::vector<double>& result, std::size_t size, std::size_t row_count)
{
    for (std::size_t i = 0; i < row_count; ++i) {
        for (std::size_t k = 0; k < size; ++k) {
            const double value = a[i * size + k];
            for (std::size_t j = 0; j < size; ++j) {
                result[i * size + j] += value * b[k * size + j];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int input_ok = 1;
    std::string error_message;
    matrix a;
    matrix b;
    if (rank == 0) {
        try {
            if (argc != 4) throw std::runtime_error("Использование: lab3 A.txt B.txt C.txt");
            a = read_matrix(argv[1]);
            b = read_matrix(argv[2]);
            if (a.size != b.size) throw std::runtime_error("Размеры матриц не совпадают.");
            if (a.size > static_cast<std::size_t>(std::numeric_limits<int>::max()) / a.size) {
                throw std::runtime_error("Матрица слишком велика для счётчиков MPI.");
            }
        }
        catch (const std::exception& error) {
            input_ok = 0;
            error_message = error.what();
        }
    }

    MPI_Bcast(&input_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!input_ok) {
        if (rank == 0) std::cerr << "Ошибка: " << error_message << '\n';
        MPI_Finalize();
        return 1;
    }

    unsigned long long size_value = rank == 0 ? static_cast<unsigned long long>(a.size) : 0;
    MPI_Bcast(&size_value, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    const std::size_t n = static_cast<std::size_t>(size_value);
    const int element_count = static_cast<int>(n * n);

    std::vector<double> b_values(n * n);
    if (rank == 0) b_values = std::move(b.values);

    std::vector<int> counts(static_cast<std::size_t>(process_count));
    std::vector<int> displacements(static_cast<std::size_t>(process_count));
    for (int process = 0; process < process_count; ++process) {
        const std::size_t first_row = n * static_cast<std::size_t>(process) / static_cast<std::size_t>(process_count);
        const std::size_t end_row = n * static_cast<std::size_t>(process + 1) / static_cast<std::size_t>(process_count);
        counts[static_cast<std::size_t>(process)] = static_cast<int>((end_row - first_row) * n);
        displacements[static_cast<std::size_t>(process)] = static_cast<int>(first_row * n);
    }

    const std::size_t local_first_row = n * static_cast<std::size_t>(rank) / static_cast<std::size_t>(process_count);
    const std::size_t local_end_row = n * static_cast<std::size_t>(rank + 1) / static_cast<std::size_t>(process_count);
    const std::size_t local_rows = local_end_row - local_first_row;
    const int local_element_count = counts[static_cast<std::size_t>(rank)];
    std::vector<double> local_a(local_rows * n);
    std::vector<double> local_result(local_rows * n, 0.0);
    std::vector<double> result_values(rank == 0 ? n * n : 0);

    double unused_value = 0.0;
    double* a_receive_buffer = local_a.empty() ? &unused_value : local_a.data();
    double* result_receive_buffer = rank == 0 ? result_values.data() : &unused_value;

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    MPI_Bcast(b_values.data(), element_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? a.values.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 a_receive_buffer, local_element_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    multiply_rows(local_a, b_values, local_result, n, local_rows);

    MPI_Gatherv(local_result.empty() ? &unused_value : local_result.data(), local_element_count, MPI_DOUBLE,
                result_receive_buffer, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;

    int output_ok = 1;
    if (rank == 0) {
        try {
            matrix result;
            result.size = n;
            result.values = std::move(result_values);
            write_matrix(result, argv[3]);

            std::cout << "size=" << n << '\n';
            std::cout << "time_seconds=" << std::fixed << std::setprecision(9) << elapsed << '\n';
            std::cout << "matrix_bytes=" << 3 * n * n * sizeof(double) << '\n';
            std::cout << "scalar_operations=" << std::fixed << std::setprecision(0) << 2.0L * n * n * n << '\n';
            std::cout << "processes=" << process_count << '\n';
        }
        catch (const std::exception& error) {
            output_ok = 0;
            error_message = error.what();
        }
    }
    MPI_Bcast(&output_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!output_ok && rank == 0) std::cerr << "Ошибка: " << error_message << '\n';

    MPI_Finalize();
    return output_ok ? 0 : 1;
}
