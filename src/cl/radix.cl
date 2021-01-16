#define LOCAL_SIZE 128


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