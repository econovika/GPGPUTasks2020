#define TILE_SIZE 16

__kernel void matrix_multiplication(const __global float *const a,
                                    const __global float *const b,
                                    __global float *const c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N
                                   )
{
    unsigned int y = get_global_id(0); // number of col in C (from 0 to N)
    unsigned int x = get_global_id(1); // number of row in C (from 0 to M)

    if (y >= N || x >= M)
        return;

    unsigned int loc_y = get_local_id(0); // number of col in tile (from 0 to TILE_SIZE)
    unsigned int loc_x = get_local_id(1); // number of row in tile (from 0 to TILE_SIZE)

    __local float tileA[TILE_SIZE * (TILE_SIZE + 1)];
    __local float tileB[TILE_SIZE * (TILE_SIZE + 1)];

    float sum = 0.0f;
    for (size_t tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[loc_x * (TILE_SIZE + 1) + loc_y] = a[x * K + (tileK * TILE_SIZE + loc_y)];
        tileB[loc_x * (TILE_SIZE + 1) + loc_y] = b[y + (tileK * TILE_SIZE + loc_x) * N];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t k = 0; k < TILE_SIZE; ++k)
            sum += tileA[loc_x * (TILE_SIZE + 1) + k] * tileB[k * (TILE_SIZE + 1) + loc_y];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[x * N + y] = sum;
}