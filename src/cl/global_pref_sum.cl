#define LOCAL_SIZE 128


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

    if ((group_id >> pow) & 1) // if 2^pow in group_id decomposition
        // how many 0s before given group
        sum[group_id] += ys[group_id / (1 << pow) - 1]
}