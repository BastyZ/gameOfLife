#include <iostream>
#include <algorithm>
#include <chrono>

using namespace std;
#define N 1024
#define RADIUS 3
#define BLOCK_SIZE 16

typedef unsigned char ubyte;

ubyte* m_data, *m_resultData;
ubyte* d_m_data, *d_m_resultData;

cudaError_t err1;
cudaError_t err2;
int EXIT_ERROR = 2;

ushort threadsCount = 1024;

// game of life settings
uint bitLifeBytesPerTrhead = 1u;

size_t m_worldWidth = 20000;
size_t m_worldHeight = 20000;
size_t m_dataLength;  // m_worldWidth * m_worldHeight
size_t lifeIteratinos = 1000;

// ---

__global__ void simpleLifeKernel(const ubyte *lifeData, uint worldWidth,
                                 uint worldHeight, ubyte *resultLifeData) {
    uint worldSize = worldWidth * worldHeight;

    for (uint cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
         cellId < worldSize;
         cellId += blockDim.x * gridDim.x) {
        uint x = cellId % worldWidth;
        uint yAbs = cellId - x;
        uint xLeft = (x + worldWidth - 1) % worldWidth;
        uint xRight = (x + 1) % worldWidth;
        uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
        uint yAbsDown = (yAbs + worldWidth) % worldSize;

        uint aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp]
                          + lifeData[xRight + yAbsUp] + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
                          + lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

        resultLifeData[x + yAbs] =
                aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
    }
}

void runSimpleLifeKernel(ubyte *&d_lifeData, ubyte *&d_lifeDataBuffer, size_t worldWidth,
                         size_t worldHeight, size_t iterationsCount, ushort threadsCount) {
    //assert((worldWidth * worldHeight) % threadsCount == 0);
    size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
    ushort blocksCount = (ushort) std::min((size_t) 32768, reqBlocksCount);

    for (size_t i = 0; i < iterationsCount; ++i) {
        simpleLifeKernel << < blocksCount, threadsCount >> > (d_lifeData, worldWidth,
                worldHeight, d_lifeDataBuffer);
        std::swap(d_lifeData, d_lifeDataBuffer);
    }
}

int main(){
    m_dataLength=m_worldWidth*m_worldHeight;
    int size = m_dataLength*sizeof(ubyte);

    m_data=(ubyte*) malloc(size);
    m_resultData=(ubyte*) malloc(size);

    err1 = cudaMalloc((void**)&d_m_data, size);
    err2 =cudaMalloc((void**)&d_m_resultData, size);

    if( err1 != cudaSuccess ) {
        printf("CUDA error: %s\n", cudaGetErrorString(err1));
        return EXIT_ERROR;
    }

    if( err2 != cudaSuccess ) {
        printf("CUDA error: %s\n", cudaGetErrorString(err2));
        return EXIT_ERROR;
    }

    for(int i=0;i<m_dataLength;i++){
        m_data[i] = (ubyte) rand() % 2;
    }

    cudaMemcpy(d_m_data, m_data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_resultData, m_resultData, size, cudaMemcpyHostToDevice);

    clock_t start = clock();
    runSimpleLifeKernel(d_m_data, d_m_resultData, m_worldWidth, m_worldHeight, lifeIteratinos, threadsCount);
    clock_t finish = clock();
    std::cout << (double(finish - start) / CLOCKS_PER_SEC) << std::endl ;
}
