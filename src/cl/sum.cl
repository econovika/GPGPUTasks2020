#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_v1(__global const unsigned int* xs,
                     __global unsigned int*       res,
                     unsigned int                 n)
{
    int id = get_global_id(0);
    if (id >= n)
        return;
    atomic_add(res, xs[id]);
}