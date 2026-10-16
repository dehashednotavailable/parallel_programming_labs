#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdlib>
#include <limits.h>
#include <limits>
#include <sys/time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct matrix
{
    std::size_t size;
    std::vector<double> values;
};

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

matrix read_matrix(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    if (!file) throw std::runtime_error("Не удалось открыть файл: " + filename);

    unsigned long long size_value = 0;
    if (!(file >> size_value) || size_value == 0
        || size_value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Неверный размер матрицы: " + filename);
    }

    matrix result;
    result.size = static_cast<std::size_t>(size_value);
    if (result.size > static_cast<std::size_t>(INT_MAX) / result.size) {
        throw std::runtime_error("Слишком большая матрица для демонстрационной программы.");
    }

    for (std::size_t i = 0; i < result.size * result.size; ++i) {
        double value = 0.0;
        if (!(file >> value) || value != value || value > DBL_MAX || value < -DBL_MAX) {
            throw std::runtime_error("Недостаточно элементов или неверное число: " + filename);
        }
        result.values.push_back(value);
    }

    std::string extra;
    if (file >> extra) throw std::runtime_error("Лишние данные в матрице: " + filename);
    return result;
}

void write_matrix(const matrix& result, const std::string& filename)
{
    std::ofstream file(filename.c_str());
    if (!file) throw std::runtime_error("Не удалось создать файл: " + filename);

    file << result.size << '\n' << std::setprecision(17);
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

template <int TILE_SIZE>
__global__ void multiply_matrix_kernel(const double* a, const double* b, double* result, int size)
{
    __shared__ double tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_b[TILE_SIZE][TILE_SIZE];

    const int row = static_cast<int>(blockIdx.y) * TILE_SIZE + static_cast<int>(threadIdx.y);
    const int column = static_cast<int>(blockIdx.x) * TILE_SIZE + static_cast<int>(threadIdx.x);
    const int tile_row = static_cast<int>(threadIdx.y);
    const int tile_column = static_cast<int>(threadIdx.x);
    double value = 0.0;

    for (int start = 0; start < size; start += TILE_SIZE) {
        const int a_column = start + tile_column;
        const int b_row = start + tile_row;
        tile_a[tile_row][tile_column] = row < size && a_column < size ? a[row * size + a_column] : 0.0;
        tile_b[tile_row][tile_column] = b_row < size && column < size ? b[b_row * size + column] : 0.0;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) value += tile_a[tile_row][k] * tile_b[k][tile_column];
        __syncthreads();
    }

    if (row < size && column < size) result[row * size + column] = value;
}

template <int TILE_SIZE>
void run_kernel(const double* a, const double* b, double* result, int size)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((size + TILE_SIZE - 1) / TILE_SIZE, (size + TILE_SIZE - 1) / TILE_SIZE);
    multiply_matrix_kernel<TILE_SIZE><<<grid, block>>>(a, b, result, size);
}

void launch_kernel(int tile_size, const double* a, const double* b, double* result, int size)
{
    switch (tile_size) {
        case 4: run_kernel<4>(a, b, result, size); break;
        case 8: run_kernel<8>(a, b, result, size); break;
        case 16: run_kernel<16>(a, b, result, size); break;
        case 32: run_kernel<32>(a, b, result, size); break;
        default: throw std::runtime_error("Размер блока должен быть 4, 8, 16 или 32.");
    }
}

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Использование: " << argv[0] << " A.txt B.txt C.txt размер_блока\n";
        return 1;
    }

    try {
        const int tile_size = std::atoi(argv[4]);
        if (tile_size != 4 && tile_size != 8 && tile_size != 16 && tile_size != 32) {
            throw std::runtime_error("Размер блока должен быть 4, 8, 16 или 32.");
        }

        const matrix a = read_matrix(argv[1]);
        const matrix b = read_matrix(argv[2]);
        if (a.size != b.size) throw std::runtime_error("Размеры матриц не совпадают.");

        cudaDeviceProp device;
        check_cuda(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
        check_cuda(cudaSetDevice(0), "cudaSetDevice");

        const std::size_t element_count = a.size * a.size;
        const std::size_t bytes = element_count * sizeof(double);
        std::vector<double> result_values(element_count, 0.0);
        double* device_a = 0;
        double* device_b = 0;
        double* device_result = 0;
        cudaEvent_t start_event;
        cudaEvent_t stop_event;
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate");

        struct timeval wall_start;
        struct timeval wall_finish;
        gettimeofday(&wall_start, 0);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes), "cudaMalloc A");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes), "cudaMalloc B");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_result), bytes), "cudaMalloc C");
        check_cuda(cudaMemcpy(device_a, &a.values[0], bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
        check_cuda(cudaMemcpy(device_b, &b.values[0], bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");
        launch_kernel(tile_size, device_a, device_b, device_result, static_cast<int>(a.size));
        check_cuda(cudaGetLastError(), "CUDA kernel launch");
        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop_event), "CUDA kernel synchronization");

        float kernel_milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&kernel_milliseconds, start_event, stop_event), "cudaEventElapsedTime");
        check_cuda(cudaMemcpy(&result_values[0], device_result, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");

        check_cuda(cudaFree(device_a), "cudaFree A");
        check_cuda(cudaFree(device_b), "cudaFree B");
        check_cuda(cudaFree(device_result), "cudaFree C");
        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");

        matrix result;
        result.size = a.size;
        result.values.swap(result_values);
        write_matrix(result, argv[3]);
        gettimeofday(&wall_finish, 0);
        const double wall_seconds = static_cast<double>(wall_finish.tv_sec - wall_start.tv_sec)
            + static_cast<double>(wall_finish.tv_usec - wall_start.tv_usec) / 1000000.0;
        const unsigned long long n = static_cast<unsigned long long>(a.size);

        std::cout << "device=" << device.name << '\n';
        std::cout << "size=" << n << '\n';
        std::cout << "tile_size=" << tile_size << '\n';
        std::cout << "time_seconds=" << std::fixed << std::setprecision(9) << kernel_milliseconds / 1000.0 << '\n';
        std::cout << "end_to_end_seconds=" << std::setprecision(6) << wall_seconds << '\n';
        std::cout << "matrix_bytes=" << 24U * n * n << '\n';
        std::cout << "scalar_operations=" << std::setprecision(0) << 2.0 * n * n * n << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
