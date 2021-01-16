#define LOCAL_SIZE 128

#define swap(a, b)                  \
{                                   \
    __local unsigned int *tmp = a;  \
    a = b;                          \
    b = tmp;                        \
}                                   \


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