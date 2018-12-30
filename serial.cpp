#include "serial.h"
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <ctime>

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

int run(int lado, int iteraciones, int work_size) {
    // settings
    m_worldWidth=lado;
    m_worldHeight=lado;
    m_dataLength=m_worldWidth*m_worldHeight;
    m_data=(ubyte*) malloc(m_dataLength*sizeof(ubyte));
    m_resultData=(ubyte*) malloc(m_dataLength*sizeof(ubyte));

    for(int i=0;i<m_dataLength;i++){
        m_data[i] = (ubyte) rand() % 2;
    }

    int n = iteraciones;

    while (n--) {
        //clock_t check = clock();
        //if ((double(check - watch) / CLOCKS_PER_SEC) >= 10) {
        //    cout << n << " in 10 secs" << endl;
        //    break;
        //}
        computeIterationSerial();
        /*    cout << "----- " << n << " ------" << endl;
            for(int i=0;i<m_worldHeight;i++){
                for(int j=0;j<m_worldWidth;j++){
                    //printf("%u ",  m_resultData[j+i*m_worldWidth]);
                    if (m_resultData[j+i*m_worldWidth] == 0) cout << "  ";
                    if (m_resultData[j+i*m_worldWidth] == 1) cout << "o ";
                }
                cout << " |" << endl << "| ";
            }
            usleep(400000); */
    }
    free(m_resultData);
    free(m_data);
}


int main(void) {
    FILE *f = fopen("serial_results.csv","w+b");

    int contador = 5;
    int size = 32;
    int iter = 100;
    while(contador--) {
        run(size, iter, 32);
    }
    fprintf(f,"lado;celdas;tiempo\n");
    printf("Running Test 1 \n");
    for(int i = 1; i<128; i++){
        clock_t t;
        t = clock();
        run(8*i,iter, 64);
        t= clock()-t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        fprintf(f,"%d;%d;%f\n",i*8,(i*8)*(i*8),time_taken);
        fflush(f);
        printf("|");
        fflush(stdout);
    }
    printf("\n");

    fclose(f);
}