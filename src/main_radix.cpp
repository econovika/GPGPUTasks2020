#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/pref_sum_cl.h"

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


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    std::vector<std::vector<unsigned int>> bs_vectors;
    std::vector<unsigned int> sum_vec(n / 128, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        //as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
        as[i] = (unsigned int) i + 1;
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sum;
    cpu_sum = as;
    const unsigned int pow = (1 << 0);
    for (int i = 0; i < n; i++)
        cpu_sum[i] = ((as[i] & pow) + 1) & 1;
    for (int i = 0; i < n; i++) {
        //if (i % 128 != 0)
            cpu_sum[i] += cpu_sum[i - 1];
    }
    printf("CPU: %i\n", cpu_sum[n - 1]);

    {
        ocl::Kernel local_pref_sum(pref_sum_kernel, pref_sum_kernel_length, "local_pref_sum");
        ocl::Kernel global_pref_sum(pref_sum_kernel, pref_sum_kernel_length, "global_pref_sum");
        ocl::Kernel radix_sort(pref_sum_kernel, pref_sum_kernel_length, "radix_sort");
        local_pref_sum.compile();
        global_pref_sum.compile();
        radix_sort.compile();

        unsigned int workGroupSize = 128;
        int pow = 0;

        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        gpu::gpu_mem_32u sum;
        sum.resizeN(n / 128);

        unsigned int k = n;



        while (k > n / 128 / 128) {
            printf("%i\n", k);
            unsigned int global_work_size = (k + workGroupSize - 1) / workGroupSize * workGroupSize;

            std::vector<unsigned int> bs(k, 0);


            gpu::gpu_mem_32u bs_gpu;
            bs_gpu.resizeN(k);

            if (k == n)
                local_pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, 0, k, 1, 0);
            else
                local_pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, 0, k, 0, 1);
            k /= workGroupSize;

            bs_gpu.readN(bs.data(), k);
            bs_vectors.push_back(bs);

//            std::vector<unsigned int> as(k, 0);
            bs_gpu.readN(as.data(), k);
            as_gpu.writeN(as.data(), k);

            global_work_size = (k + workGroupSize - 1) / workGroupSize * workGroupSize;
            global_pref_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), bs_gpu, sum, pow, k);
            pow++;
            sum.readN(sum_vec.data(), k);
//            printf("%i\t%i\n", as[0], k);
        }
    }
    //for (int i = 0; i < 128; i++)
    printf("CPU: %i\n", cpu_sum[n - 1]);
    int summa = 0;
    for (int i = 0; i < n / 128 / 128 / 128; i++)
        summa += bs_vectors[2][0];
    printf("GPU: %i\n", summa);
    printf("ELEMENTS BEFORE GROUP: %i", sum_vec[1]);
    return 0;
}
