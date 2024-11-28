#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Handling arbitrary vector size
  if (tid < n) {
    out[tid] = a[tid] + b[tid];
  }
}

int main(){
  
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  
  printf("Number of devices: %d\n", nDevices);
  
  for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
      printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
      printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
      printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
      printf("  minor-major: %d-%d\n", prop.minor, prop.major);
      printf("  Warp-size: %d\n", prop.warpSize);
      printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
      printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
    }

    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    printf("a[0] = %f\n", a[0]);
    printf("b[0] = %f\n", b[0]);

    // Executing kernel 
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    vector_add<<<grid_size, block_size>>>(d_out, d_a, d_b, N);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("out[0] = %f\n", out[0]);

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
