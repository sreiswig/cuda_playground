#include <iostream>
#include <bits/stdc++.h>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vector_diff(float *out, float *a, float *b, int n) {
  for(int i = 0; i < n; i++) {
    out[i] = a[i] - b[i];
  }
}

int main() {
  std::string line;
  std::vector<float> list1;
  std::vector<float> list2;
  std::ifstream input ("input.txt");

  if (input.is_open()) {
    while (std::getline(input, line, ' ')) {
      list1.push_back(std::stof(line));
      std::getline(input, line);
      list2.push_back(std::stof(line));
    }
    input.close();
  }

  int elements = list1.size();
  float *a = list1.data();
  float *b = list2.data();
  float *out = (float*)malloc(sizeof(float) * elements);

  float *d_a, *d_b, *d_out;

  cudaMalloc((void**)&d_a, sizeof(float) * elements);
  cudaMalloc((void**)&d_b, sizeof(float) * elements);
  cudaMalloc((void**)&d_out, sizeof(float) * elements);

  cudaMemcpy(d_a, a, sizeof(float) * elements, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * elements, cudaMemcpyHostToDevice);

  vector_diff<<<1,1>>>(d_out, d_a, d_b, elements);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(out, d_out, sizeof(float) * elements, cudaMemcpyDeviceToHost);
  printf("out[0] = %f\n", out[0]);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  free(out);
}
