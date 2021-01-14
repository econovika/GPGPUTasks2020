#define LOCAL_SIZE 128

#define swap(a, b)                                                             \
  {                                                                            \
    __local unsigned int *tmp = a;                                             \
    a = b;                                                                     \
    b = tmp;                                                                   \
  }                                                                            \

__kernel void local_pref_sum(__global unsigned int* xs,
                             __global unsigned int* ys,
                             const int bit,
                             const int n,
                             const int glob,
                             const int no_bit)
{
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int group_size = get_local_size(0);
    const int global_id = group_id*group_size + local_id;

    if (global_id >= n)
        return;

    __local unsigned int a[LOCAL_SIZE];
    __local unsigned int b[LOCAL_SIZE];

    __local unsigned int* buf_a = a;
    __local unsigned int* buf_b = b;

    if (no_bit == 0) {
        const unsigned int pow = (1 << bit);
        buf_a[local_id] = ((xs[global_id] & pow) + 1) & 1; // reverse last bit
        buf_b[local_id] = ((xs[global_id] & pow) + 1) & 1; // reverse last bit
    }
    else {
        buf_a[local_id] = xs[global_id];
        buf_b[local_id] = xs[global_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 1; step < LOCAL_SIZE; step <<= 1)
    // compute how many 0 bits before current element
    {
        if (local_id >= step)
            buf_b[local_id] = buf_a[local_id] + buf_a[local_id - step];
        else
            buf_b[local_id] = buf_a[local_id];
        swap(buf_a, buf_b);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (glob == 1)
        // save how many 0 bits in current group
        ys[group_id] = buf_a[LOCAL_SIZE - 1];
    else
        // save how many 0 bits before current work item
        ys[global_id] = buf_a[local_id];
}


__kernel void global_pref_sum(__global unsigned int* ys,
                              __global unsigned int* sum,
                              const int pow,
                              const int n)
{
    const int global_id = get_global_id(0);
    if (global_id >= n)
        return;

    int group_id = get_group_id(0);

    if ((group_id >> pow) & 1) // 2^pow in group_id decomposition
        sum[group_id] += ys[group_id / (1 << pow) - 1];
}


__kernel void radix_sort(__global unsigned int* sum,
                         __global unsigned int* xs,
                         __global unsigned int* ys)
{
    const int global_id = get_global_id(0);
    ys[sum[global_id]] = xs[global_id];
}

__kernel void sum(__global const unsigned int* xs,
                     __global unsigned int*       ys,
                     unsigned int                 n,
                     int s)
{
    __local unsigned int tmp[LOCAL_SIZE];
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);

    tmp[local_id] = xs[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id % (2*s) == 0) {
        tmp[local_id] += tmp[local_id + s];
        ys[global_id] = tmp[local_id];
    }
}