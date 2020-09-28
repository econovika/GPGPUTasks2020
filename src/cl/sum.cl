#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_v2(__global const unsigned int* xs,
                     __global unsigned int*       res,
                     __local  unsigned int*       l_xs,
                     unsigned int                 n)
{
    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    int groupSize = get_local_size(0);

    int idx = groupId * groupSize * 2 + localId;

    // init local memory (half of threads on 1 iter are idle, add values of next group also)
    l_xs[localId] = idx < n ? xs[idx] + xs[idx + groupSize] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // sequential addressing
    for (int step = groupSize / 2; step > 32; step >>= 1) {
        if (localId < step)
            l_xs[localId] += l_xs[localId + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // sum first warp
    // no barrier is needed
    // no need of 'if (localId < step)'
    if (localId < 32) {
        l_xs[localId] += l_xs[localId + 32];
        l_xs[localId] += l_xs[localId + 16];
        l_xs[localId] += l_xs[localId + 8];
        l_xs[localId] += l_xs[localId + 4];
        l_xs[localId] += l_xs[localId + 2];
        l_xs[localId] += l_xs[localId + 1];
    }

    if (localId == 0) atomic_add(res, l_xs[0]);
}
