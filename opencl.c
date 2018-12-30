#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <unistd.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)

// a: ancho/alto de la matriz | i: iteraciones
int run(int a, int i, int c) {
    // Create the two input vectors
    size_t *m_worldWidth = (size_t *) malloc(sizeof(size_t));
    size_t *m_worldHeight = (size_t *) malloc(sizeof(size_t));

    m_worldWidth[0] = a;
    m_worldHeight[0] = a;
    time_t t;
    srand((unsigned) time(&t));

    size_t m_dataLength = m_worldWidth[0] * m_worldHeight[0];

    size_t lifeIteratinos = i;

    const int LIST_SIZE = m_dataLength;
    char *A = (char *) malloc(sizeof(char) * LIST_SIZE);
    char *B = (char *) malloc(sizeof(char) * LIST_SIZE);
    for (int i = 0; i < LIST_SIZE; i++) {
        A[i] = rand() % 2;
    }
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("lifeKernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
                         &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      LIST_SIZE * sizeof(char), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      LIST_SIZE * sizeof(char), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(size_t), NULL, &ret);
    cl_mem d_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(size_t), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                               sizeof(size_t), m_worldWidth, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_mem_obj, CL_TRUE, 0,
                               sizeof(size_t), m_worldHeight, 0, NULL, NULL);
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **) &source_str, (const size_t *) &source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "lifeKernel", &ret);


    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = c; // Process in groups of 64

    while(lifeIteratinos--) {
        // Copy the lists A and B to their respective memory buffers
        ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                                   LIST_SIZE * sizeof(char), A, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                                   LIST_SIZE * sizeof(char), B, 0, NULL, NULL);

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &a_mem_obj);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &c_mem_obj);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_mem_obj);
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &b_mem_obj);

        // Execute the OpenCL kernel on the list
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                     &global_item_size, &local_item_size, 0, NULL, NULL);

        // Read the memory buffer C on the device to the local variable C
        ret = clEnqueueReadBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                                  LIST_SIZE * sizeof(char), B, 0, NULL, NULL);

        // Swap pointers
        char *aux = A;
        A = B;
        B = aux;

        // Game visualization
/*        printf("\n---------------------- %d ----------------------\n|", (int) lifeIteratinos);
        for(int i=0;i<m_worldHeight[0];i++){
            for(int j=0;j<m_worldWidth[0];j++){
                //printf("%u ",  m_resultData[j+i*m_worldWidth]);
                //printf("%d ", B[j+i*m_worldWidth[0]]);
                if (B[j+i*m_worldWidth[0]] == 0) printf("  ");
                if (B[j+i*m_worldWidth[0]] == 1) printf("° ");
            }
            printf("|\n|");
        }
        usleep(100000);*/
    }

    // Display the result to the screen
    //for(i = 0; i < LIST_SIZE; i++)
    //    printf("%d\n",A[i]);
    //for(i = 0; i < LIST_SIZE; i++)
    //    printf("%d\n",A[i]);
/*    for(int i=0;i<m_worldHeight[0];i++){
        for(int j=0;j<m_worldWidth[0];j++){
            //printf("%u ",  m_resultData[j+i*m_worldWidth]);
            //printf("%d ", A[j+i*m_worldWidth[0]]);
            if (A[j+i*m_worldWidth[0]] == 0) printf("  ");
            if (A[j+i*m_worldWidth[0]] == 1) printf("° ");
        }
        printf("\n");
    }
    printf("------- B ------\n");
    for(int i=0;i<m_worldHeight[0];i++){
        for(int j=0;j<m_worldWidth[0];j++){
            //printf("%u ",  m_resultData[j+i*m_worldWidth]);
            //printf("%d ", B[j+i*m_worldWidth[0]]);
            if (B[j+i*m_worldWidth[0]] == 0) printf("  ");
            if (B[j+i*m_worldWidth[0]] == 1) printf("° ");
        }
        printf("\n");
    }*/
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseMemObject(d_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(m_worldWidth);
    free(m_worldHeight);
    free(source_str);
    return 0;
}

int main(void) {

    FILE *f = fopen("aaaopencl_results.csv","w+b");

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
        run(8*i,iter, 47);
        t= clock()-t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        fprintf(f,"%d;%d;%f\n",i*8,(i*8)*(i*8),time_taken);
        fflush(f);
        printf("|");
        fflush(stdout);
    }
    printf("\n");
}
