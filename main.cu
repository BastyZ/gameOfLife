#include <iostream>
#include <algorithm>

using namespace std;
#define N 1024
#define RADIUS 3
#define BLOCK_SIZE 16

typedef unsigned char ubyte;

ubyte* m_data;
ubyte* m_resultData;

size_t m_worldWidth;
size_t m_worldHeight;
size_t m_dataLength;  // m_worldWidth * m_worldHeight

// Maybe useful data

int screenWidth = 1024;
int screenHeight = 768;

ushort threadsCount = 256;

// game of life settings
size_t lifeIteratinos = 1;
uint bitLifeBytesPerTrhead = 1u;

size_t worldWidth = 256;
size_t worldHeight = 256;

size_t newWorldWidth = 256;
size_t newWorldHeight = 256;

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
    m_worldWidth=20;
    m_worldHeight=20;
    m_dataLength=m_worldWidth*m_worldHeight;
    m_data=(ubyte*) malloc(m_dataLength*sizeof(ubyte));
    m_resultData=(ubyte*) malloc(m_dataLength*sizeof(ubyte));
    for(int i=0;i<m_dataLength;i++){
        m_data[i] = (ubyte) rand() % 2;
    }
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            printf("%u ",  m_data[j+i*10]);
        }
        cout << endl;
    }
    int n = 1000;
    while (n--) {
        computeIterationSerial();
        cout << "-----" << n << "------" << endl;
        for(int i=0;i<m_worldHeight;i++){
            for(int j=0;j<m_worldWidth;j++){
                printf("%u ",  m_data[j+i*10]);
            }
            cout << endl;
        }
        usleep(500000);
    }
}
