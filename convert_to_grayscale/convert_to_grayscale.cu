__global__
void colortoGrayscaleConversion(unsigned char * Pout,
                                unsigned char * Pin, int width, int height) {
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    int grayOffset = row*width + col;

    int rgbOffset = grayOffset*CHANNELS;
    unsigned char r = Pin[rgbOffset];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];

    Pout[grayOffset] = 0.21*r + 0.71*g + 0.07*b;
  }
}
