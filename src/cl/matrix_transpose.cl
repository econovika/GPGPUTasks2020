#define TILE_SIZE 16

__kernel void matrix_transpose(const __global float *const m,
                               __global float *mt,
                               const unsigned int h,
                               const unsigned int w)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    // declare local array to be shared by all work items within a group => groupSize
    __local float tile[TILE_SIZE * (TILE_SIZE + 1)];

    const unsigned int local_x = get_local_id(0); // in group
    const unsigned int local_y = get_local_id(1);

    // from global memory to local
    if (x < w && y < h)
        tile[local_y * (TILE_SIZE + 1) + local_x] = m[x * h + y];
    barrier(CLK_LOCAL_MEM_FENCE);

    // write transposed to global memory
    if (x < w && y < h)
        mt[y * w + x] = tile[local_y * (TILE_SIZE + 1) + local_x];
}