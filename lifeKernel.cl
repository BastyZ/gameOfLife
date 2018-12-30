__kernel void lifeKernel(__global char *lifeData,__global size_t* worldWidth,
                                 __global size_t* worldHeight, __global char *resultLifeData) {
    int worldSize = worldWidth[0]* worldHeight[0];
    // Get the index of the current element

    for (int cellId = get_global_id(0);
        cellId < worldSize;
        cellId += get_local_size(0)) {

        int x = cellId % worldWidth[0];
        int yAbs = cellId - x;
        int xLeft = (x + worldWidth[0] - 1) % worldWidth[0];
        int xRight = (x + 1) % worldWidth[0];
        int yAbsUp = (yAbs + worldSize - worldWidth[0]) % worldSize;
        int yAbsDown = (yAbs + worldWidth[0]) % worldSize;

        int aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp]
                          + lifeData[xRight + yAbsUp] + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
                          + lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];
      resultLifeData[x + yAbs] =
              aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
    }
}