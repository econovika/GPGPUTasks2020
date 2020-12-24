#include clion_defines.cl


__kernel void matrix_transpose(__global const *m,
                               __global const *mt,
                               const unsigned int m,
                               const unsigned int k)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    __local float tile[WARP_SIZE][WARP_SIZE + 1];
    const unsigned int loc_i = get_local_id(0);
    const unsigned int loc_j = get_local_id(1);

    tile[j * WARP_SIZE][i] = m[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE)

    const float tmp = tile[j * WARP_SIZE][i];
    tile[j * WARP_SIZE][i] = tile[i * WARP_SIZE][j];
    tile[i * WARP_SIZE][j] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE)

    mt[j * m + i] = tile[i * WARP_SIZE][j];
}