#define LOCAL_SIZE 128

#define swap(a, b)                  \
{                                   \
    __local unsigned int *tmp = a;  \
    a = b;                          \
    b = tmp;                        \
}                                   \


__kernel void radix(__global unsigned int* sum,
                    __global unsigned int* xs,
                    __global unsigned int* ys,
                    const int n)
{
    const int global_id = get_global_id(0);
    if (global_id >= n)
        return;
    // sum[global_id] - how many 0s in massive before given global_id
    ys[sum[global_id]] = xs[global_id];
}

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

__kernel void global_pref_sum(__global unsigned int* ys,
                              __global unsigned int* sum,
                              const int pow,
                              const int n)
{
    // Kernel to compute pow_th pref sums for given group
    const int global_id = get_global_id(0);
    if (global_id >= n)
        return;
    const int group_id = get_group_id(0);
    const int local_id = get_local_id(0);

    if (( (group_id >> pow) & 1 ) && ( local_id % (2 * step) == 0 )) // if 2^pow in group_id decomposition
        // how many 0s before given group
        sum[group_id] += ys[group_id / (1 << pow) - 1]
}

__kernel void local_pref_sum(__global unsigned int* xs,
                             __global unsigned int* ys,
                             __global unsigned int* sum,
                             const int n,
                             const int bit,
                             const int global)
{
    // Kernel to sum how many member numbers before i_th
    // have 0s in bit_th bit

    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int global_id = group_id * LOCAL_SIZE + local_id;

    if (global_id >= n)
        return;

    __local unsigned int a[LOCAL_SIZE];
    __local unsigned int b[LOCAL_SIZE];

    __local unsigned int* buf_a = a;
    __local unsigned int* buf_b = b;

    const unsigned int pow = (1 << bit);
    buf_a[local_id] = ((xs[global_id] & pow) + 1) & 1; // reverse last bit
    buf_b[local_id] = ((xs[global_id] & pow) + 1) & 1; // reverse last bit

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 1; step < LOCAL_SIZE; step <<= 1)
    {
        if (local_id >= step)
            buf_b[local_id] = buf_a[local_id] + buf_a[local_id - step];
        else
            buf_b[local_id] = buf_a[local_id];
        swap(buf_a, buf_b);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global == 1)
        ys[group_id] = buf_a[LOCAL_SIZE - 1];
    else {
        ys[global_id] = buf_a[local_id] + sum[group_id];
    }
}
