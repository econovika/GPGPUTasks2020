#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include <math.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int n =  32;
    std::vector<unsigned int> xs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        xs[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = xs;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    // Input vector with initial numbers
    gpu::gpu_mem_32u xs_gpu;
    xs_gpu.resizeN(n);

    // Output vector sorted

    {
        ocl::Kernel global_pref_sum(radix_kernel, radix_kernel_length, "global_pref_sum");
        global_pref_sum.compile();

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel local_pref_sum(radix_kernel, radix_kernel_length, "local_pref_sum");
        local_pref_sum.compile();

        ocl::Kernel part_sum(radix_kernel, radix_kernel_length, "part_sum");
        part_sum.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            xs_gpu.writeN(xs.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 16;
            unsigned int global_work_size = std::ceil(n / workGroupSize) * workGroupSize;

            // Prefix sums before each group
            gpu::gpu_mem_32u global_sum_gpu;
            global_sum_gpu.resizeN(std::ceil(n / workGroupSize));

            for (int bit = 0; bit < 32; bit++)
            {
                gpu::gpu_mem_32u ys_gpu;
                ys_gpu.resizeN(n);

                gpu::gpu_mem_32u local_pref_sum_gpu;
                local_pref_sum_gpu.resizeN(std::ceil(n / workGroupSize));

                gpu::gpu_mem_32u sum_gpu;
                sum_gpu.resizeN(n);

                int global = 1;
                // Compute sums in each work group
                local_pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                    xs_gpu, local_pref_sum_gpu, global_sum_gpu, n, bit, global);

                int pow = 0;
                for (int step = 1; step < workGroupSize; step <<= 1)
                {
                    gpu::gpu_mem_32u part_sum_gpu;
                    part_sum_gpu.resizeN(std::ceil(n / workGroupSize));

                    // Compute partial sums
                    int work_size = std::ceil(n / workGroupSize);
                    int global_n = std::ceil(n / workGroupSize);

                    part_sum.exec(gpu::WorkSize(workGroupSize, work_size),
                                  local_pref_sum_gpu, part_sum_gpu, global_n, step);

                    // Compute global sums
                    global_pref_sum.exec(gpu::WorkSize(workGroupSize, work_size),
                                         part_sum_gpu, global_sum_gpu, pow, step, global_n);

                    pow++;
                }
                global = 0;
                local_pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                    xs_gpu, sum_gpu, global_sum_gpu, n, bit, global);

                radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                           sum_gpu, xs_gpu, ys_gpu, n);

                ys_gpu.writeN(xs.data(), n);
                xs_gpu.readN(xs.data(), n);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
        xs_gpu.readN(xs.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        printf("GPU: %i\tCPU: %i\n", xs[i], cpu_sorted[i]);
    }

    return 0;
}