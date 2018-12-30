#include <iostream>
#include <algorithm>
#include <chrono>

using namespace std;
#define N 1024
#define RADIUS 3
#define BLOCK_SIZE 64

typedef unsigned char ubyte;

ubyte* m_data, *m_resultData;
ubyte* d_m_data, *d_m_resultData;

cudaError_t err1;
cudaError_t err2;
int EXIT_ERROR = 2;

ushort threadsCount = 1024;

// game of life settings
uint bitLifeBytesPerTrhead = 1u;

size_t m_worldWidth;
size_t m_worldHeight;
size_t m_dataLength;  // m_worldWidth * m_worldHeight

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

int run(int lado, int iteraciones, int work_size) {
    // settings
    size_t m_worldWidth = lado;
    size_t m_worldHeight = lado;
    size_t lifeIteratinos = iteraciones;


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

    runSimpleLifeKernel(d_m_data, d_m_resultData, m_worldWidth, m_worldHeight, lifeIteratinos, threadsCount);
    free(m_resultData);
    free(m_data);
    return 0;
}


int main(void) {
    FILE *f = fopen("CUDA_results.csv","w+b");

    int contador = 5;
    int size = 32;
    int iter = 100;
    while(contador--) {
        run(size, iter, 32);
    }
    fprintf(f,"lado;celdas;tiempo\n");
    printf("Running Test 1: \n");
    for(int i = 1; i<128; i++){
        clock_t t;
        t = clock();
        run(8*i,iter, 64);
        t= clock()-t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        fprintf(f,"%d;%d;%f\n",i*8,(i*8)*(i*8),time_taken);
        fflush(f);
        printf("|");
    }
    printf("\n");

    fclose(f);
}
