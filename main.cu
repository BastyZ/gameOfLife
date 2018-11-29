#include <iostream>
#include <algorithm>

using namespace std;
#define N 1024
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void mykernel(void) {
}

int main(void) {
    mykernel<<<1,1>>>();
    printf("Hello World!\n");
    return 0;
}