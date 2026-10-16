#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct image_parameters
{
    float x0 = -2.0f;
    float x1 = 1.0f;
    float y0 = -1.0f;
    float y1 = 1.0f;
    int width = 1600;
    int height = 1200;
    int max_iterations = 256;
};

struct worker_parameters
{
    const image_parameters* image = nullptr;
    std::vector<int>* output = nullptr;
    int thread_id = 0;
    int thread_count = 1;
    bool cyclic_partition = true;
    double elapsed_seconds = 0.0;
};

int mandelbrot(float x, float y, int max_iterations)
{
    float z_re = x;
    float z_im = y;
    int iterations = 0;

    for (; iterations < max_iterations; ++iterations) {
        if (z_re * z_re + z_im * z_im > 4.0f) break;
        const float new_re = z_re * z_re - z_im * z_im;
        const float new_im = 2.0f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }

    return iterations;
}

void render_row(const image_parameters& image, int row, std::vector<int>& output)
{
    const float dx = (image.x1 - image.x0) / image.width;
    const float dy = (image.y1 - image.y0) / image.height;
    const float y = image.y0 + row * dy;

    for (int column = 0; column < image.width; ++column) {
        const float x = image.x0 + column * dx;
        output[static_cast<std::size_t>(row) * image.width + column] = mandelbrot(x, y, image.max_iterations);
    }
}

void render_serial(const image_parameters& image, std::vector<int>& output)
{
    for (int row = 0; row < image.height; ++row) render_row(image, row, output);
}

void worker_thread_start(worker_parameters& args)
{
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    if (args.cyclic_partition) {
        // Каждому потоку достаются строки row = thread_id + k * thread_count.
        for (int row = args.thread_id; row < args.image->height; row += args.thread_count) {
            render_row(*args.image, row, *args.output);
        }
    }
    else {
        const int first_row = args.image->height * args.thread_id / args.thread_count;
        const int last_row = args.image->height * (args.thread_id + 1) / args.thread_count;
        for (int row = first_row; row < last_row; ++row) {
            render_row(*args.image, row, *args.output);
        }
    }

    const std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();
    args.elapsed_seconds = std::chrono::duration<double>(finish - start).count();
}

void render_threaded(const image_parameters& image, std::vector<int>& output,
                     int thread_count, bool cyclic_partition,
                     std::vector<double>& worker_times)
{
    std::vector<worker_parameters> arguments(static_cast<std::size_t>(thread_count));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count - 1));

    for (int i = 0; i < thread_count; ++i) {
        arguments[static_cast<std::size_t>(i)].image = &image;
        arguments[static_cast<std::size_t>(i)].output = &output;
        arguments[static_cast<std::size_t>(i)].thread_id = i;
        arguments[static_cast<std::size_t>(i)].thread_count = thread_count;
        arguments[static_cast<std::size_t>(i)].cyclic_partition = cyclic_partition;
    }

    for (int i = 1; i < thread_count; ++i) {
        workers.emplace_back(worker_thread_start, std::ref(arguments[static_cast<std::size_t>(i)]));
    }
    worker_thread_start(arguments[0]);
    for (std::thread& worker : workers) worker.join();

    for (int i = 0; i < thread_count; ++i) {
        worker_times[static_cast<std::size_t>(i)] = arguments[static_cast<std::size_t>(i)].elapsed_seconds;
    }
}

void write_image(const image_parameters& image, const std::vector<int>& values, const std::string& filename)
{
    std::ofstream file(filename.c_str(), std::ios::binary);
    if (!file) throw std::runtime_error("Не удалось создать изображение: " + filename);

    file << "P5\n" << image.width << ' ' << image.height << "\n255\n";
    for (int count : values) {
        const unsigned char shade = static_cast<unsigned char>(255 - count * 255 / image.max_iterations);
        file.write(reinterpret_cast<const char*>(&shade), 1);
    }
    if (!file) throw std::runtime_error("Ошибка записи изображения: " + filename);
}

void parse_arguments(int argc, char* argv[], int& view, int& thread_count,
                     int& repeat_count, bool& cyclic_partition,
                     std::string& image_prefix)
{
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--view" && i + 1 < argc) view = std::stoi(argv[++i]);
        else if (argument == "--threads" && i + 1 < argc) thread_count = std::stoi(argv[++i]);
        else if (argument == "--repeats" && i + 1 < argc) repeat_count = std::stoi(argv[++i]);
        else if (argument == "--partition" && i + 1 < argc) {
            const std::string partition = argv[++i];
            if (partition == "cyclic") cyclic_partition = true;
            else if (partition == "blocks") cyclic_partition = false;
            else throw std::runtime_error("Разбиение должно быть cyclic или blocks.");
        }
        else if (argument == "--image-prefix" && i + 1 < argc) image_prefix = argv[++i];
        else throw std::runtime_error("Использование: mandelbrot --view 1|2 --threads 1..16 "
                                      "[--partition cyclic|blocks] [--repeats N] [--image-prefix путь]");
    }

    if (view != 1 && view != 2) throw std::runtime_error("Номер вида должен быть 1 или 2.");
    if (thread_count < 1 || thread_count > 16) throw std::runtime_error("Число потоков должно быть от 1 до 16.");
    if (repeat_count < 1 || repeat_count > 20) throw std::runtime_error("Число повторов должно быть от 1 до 20.");
}

int main(int argc, char* argv[])
{
    try {
        int view = 1;
        int thread_count = 2;
        int repeat_count = 5;
        bool cyclic_partition = true;
        std::string image_prefix;
        parse_arguments(argc, argv, view, thread_count, repeat_count, cyclic_partition, image_prefix);

        image_parameters image;
        if (view == 2) {
            const float scale = 0.015f;
            const float shift_x = -0.986f;
            const float shift_y = 0.30f;
            image.x0 = image.x0 * scale + shift_x;
            image.x1 = image.x1 * scale + shift_x;
            image.y0 = image.y0 * scale + shift_y;
            image.y1 = image.y1 * scale + shift_y;
        }

        const std::size_t pixel_count = static_cast<std::size_t>(image.width) * image.height;
        std::vector<int> serial_pixels(pixel_count);
        std::vector<int> threaded_pixels(pixel_count);
        std::vector<double> worker_times(static_cast<std::size_t>(thread_count));
        std::vector<double> best_worker_times;
        double serial_seconds = std::numeric_limits<double>::max();
        double threaded_seconds = std::numeric_limits<double>::max();

        for (int repeat = 0; repeat < repeat_count; ++repeat) {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            render_serial(image, serial_pixels);
            std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();
            serial_seconds = std::min(serial_seconds, std::chrono::duration<double>(finish - start).count());

            std::fill(threaded_pixels.begin(), threaded_pixels.end(), 0);
            start = std::chrono::steady_clock::now();
            render_threaded(image, threaded_pixels, thread_count, cyclic_partition, worker_times);
            finish = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(finish - start).count();
            if (elapsed < threaded_seconds) {
                threaded_seconds = elapsed;
                best_worker_times = worker_times;
            }
        }

        const bool pixels_match = serial_pixels == threaded_pixels;
        if (!image_prefix.empty()) {
            write_image(image, serial_pixels, image_prefix + "-serial.pgm");
            write_image(image, threaded_pixels, image_prefix + "-threaded.pgm");
        }

        const double worker_min = *std::min_element(best_worker_times.begin(), best_worker_times.end());
        const double worker_max = *std::max_element(best_worker_times.begin(), best_worker_times.end());
        std::cout << "view=" << view << '\n';
        std::cout << "threads=" << thread_count << '\n';
        std::cout << "partition=" << (cyclic_partition ? "cyclic" : "blocks") << '\n';
        std::cout << "repeats=" << repeat_count << '\n';
        std::cout << "serial_seconds=" << std::fixed << std::setprecision(9) << serial_seconds << '\n';
        std::cout << "parallel_seconds=" << threaded_seconds << '\n';
        std::cout << "speedup=" << serial_seconds / threaded_seconds << '\n';
        std::cout << "worker_min_seconds=" << worker_min << '\n';
        std::cout << "worker_max_seconds=" << worker_max << '\n';
        std::cout << "load_imbalance=" << (worker_min > 0.0 ? worker_max / worker_min : 0.0) << '\n';
        std::cout << "pixels_match=" << (pixels_match ? "true" : "false") << '\n';
        std::cout << "width=" << image.width << '\n';
        std::cout << "height=" << image.height << '\n';
        if (!pixels_match) return 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
