#include "serial.h"
#include <cstdlib>
#include <iostream>
#include <unistd.h>

using namespace std;
typedef unsigned char ubyte;

ubyte* m_data;
ubyte* m_resultData;

size_t m_worldWidth;
size_t m_worldHeight;
size_t m_dataLength;  // m_worldWidth * m_worldHeight


ubyte countAliveCells(ubyte* data, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1,size_t y2) {
    return data[x0 + y0] + data[x1 + y0] + data[x2 + y0]
           + data[x0 + y1] + data[x2 + y1]
           + data[x0 + y2] + data[x1 + y2] + data[x2 + y2];
}

void computeIterationSerial() {
    for (size_t y = 0; y < m_worldHeight; ++y) {
        size_t y0 = ((y + m_worldHeight - 1) % m_worldHeight) * m_worldWidth;
        size_t y1 = y * m_worldWidth;
        size_t y2 = ((y + 1) % m_worldHeight) * m_worldWidth;

        for (size_t x = 0; x < m_worldWidth; ++x) {
            size_t x0 = (x + m_worldWidth - 1) % m_worldWidth;
            size_t x2 = (x + 1) % m_worldWidth;

            ubyte aliveCells = countAliveCells(m_data,x0, x, x2, y0, y1, y2);
            m_resultData[y1 + x] =
                    aliveCells == 3 || (aliveCells == 2 && m_data[x + y1]) ? 1 : 0;
        }
    }
    std::swap(m_data, m_resultData);
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
    int n = 400;
    while (n--) {
        computeIterationSerial();
        cout << "----- " << n << " ------" << endl;
        for(int i=0;i<m_worldHeight;i++){
            for(int j=0;j<m_worldWidth;j++){
                //printf("%u ",  m_resultData[j+i*m_worldWidth]);
                if (m_resultData[j+i*m_worldWidth] == 0) cout << "  ";
                if (m_resultData[j+i*m_worldWidth] == 1) cout << "o ";
            }
            cout << " |" << endl << "| ";
        }
        usleep(400000);
    }
}