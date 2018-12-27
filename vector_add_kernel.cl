__kernel void vector_add(̣__global char *lifeData,__global size_t worldWidth,
                                 __global size_t worldHeight, __global char *resultLifeData) {
    uint worldSize = worldWidth * worldHeight;
    // Get the index of the current element
    int i = get_global_id(0);
    resultLifeData[i] = 8;
/*
    for (uint cellId = get_group_id() + get_local_id();
        cellId < worldSize;
        cellId += get_local_size()) {
        uint x = cellId % worldWidth;
        uint yAbs = cellId - x;
        uint xLeft = (x + worldWidth - 1) % worldWidth;
        uint xRight = (x + 1) % worldWidth;
        uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
        uint yAbsDown = (yAbs + worldWidth) % worldSize;

        uint aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp]
                          + lifeData[xRight + yAbsUp] + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
                          + lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

        resultLifeData[x + yAbs] = 0;
        */
        //        aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
    //}
}
