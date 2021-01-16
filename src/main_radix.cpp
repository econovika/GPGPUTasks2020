#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl/global_pref_sum_cl.h"
#include "cl/local_pref_sum_cl.h"
#include "cl/part_sum_cl.h"

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

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u ys_gpu;
    ys_gpu.resizeN(n);

    gpu::gpu_mem_32u zs_gpu;
    zs_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel global_pref_sum(global_pref_sum_kernel, global_pref_sum_kernel_length, "global_pref_sum");
        global_pref_sum.compile();

        ocl::Kernel local_pref_sum(local_pref_sum_kernel, local_pref_sum_length, "local_pref_sum");
        local_pref_sum.compile();

        ocl::Kernel part_sum(part_sum_kernel, part_sum_kernel_length, "part_sum");
        part_sum.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            gpu::gpu_mem_32u bs_gpu;
            bs_gpu.resizeN(n / workGroupSize);

            gpu::gpu_mem_32u global_sum_gpu;
            global_sum_gpu.resizeN(n / workGroupSize);

            for (int bit = 0; bit < 32; bit++)
            {
                const int global = 1;
                // Compute sums in each work group
                local_pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                    as_gpu, bs_gpu, global_sum_gpu, n, bit, global);

                int pow = 0;
                for (int step = 0; step < workGroupSize; step *= 2)
                {
                    gpu::gpu_mem_32u part_sum_gpu;
                    part_sum_gpu.resizeN(n / workGroupSize / (1 << step));

                    // Compute partial sums
                    part_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                  bs_gpu, part_sum_gpu, n / workGroupSize / (1 << step));

                    // Compute global sums
                    global_pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                         part_sum_gpu, global_sum_gpu, pow, n / workGroupSize);

                    pow++;
                }
                const int global = 0;
                local_pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                    as_gpu, ys_gpu, global_sum_gpu, n, bit, global);

                radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                           ys_gpu, as_gpu, zx_gpu, n);

                zx_gpu.readN(&as, n);
                as_gpu.writeN(as.data(), n);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}