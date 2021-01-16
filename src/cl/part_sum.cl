#define LOCAL_SIZE 128


__kernel void part_sum(__global const unsigned int* xs,
                       __global const unsigned int* ys,
                       const unsigned int n,
                       const unsigned int step)
{
    __local unsigned int tmp[LOCAL_SIZE];

    const int local_id = get_local_id(0);
    const int global_id = get_global_id(0);

    if (global_id >= n)
        return;

    tmp[local_id] = xs[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id % (2 * step) == 0)
        ys[global_id] = tmp[local_id] + tmp[local_id + step];
}