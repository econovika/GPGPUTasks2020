__kernel void radix(__global unsigned int* xs,
                   __global unsigned int* ys)
{
    int groupSize = get_local_size(0);
    int groupId = get_group_id(0);
    int localId = get_local_id(0);

    __local unsigned int tmp[groupSize] = {0};
    tmp[localId] = xs[groupId*groupSize + localId];

    for (int step = 1; step < groupSize; step *= 2) {
        if (localId >= step)
            tmp[local_id] += tmp[local_id - step];
        else
            tmp[local_id] = tmp[local_id];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    ys[groupId*groupSize + localId] = tmp[localId];
}