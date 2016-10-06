#include <stdio.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
    bool passed = true;
    for (int i = 0; i < n; i++)
        if (res[i] != ref[i]) {
          printf("%d %f %f\n", i, res[i], ref[i]);
          printf("%25s\n", "*** FAILED ***");
          passed = false;
          break;
     }

    if (passed)
        printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 / ms );
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(float *odata, const float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
        odata[x*width + (y+j)] = idata[(y+j)*width + x];
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalesced(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char **argv)
{
    const int nx = 1024;
    const int ny = 1024;
    const int mem_size = nx*ny*sizeof(float);

    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    int devId = 0;
    if (argc > 1) devId = atoi(argv[1]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);
    printf("\nDevice : %s\n", prop.name);
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    cudaSetDevice(devId);

    float *h_idata = (float*)malloc(mem_size);
    float *h_cdata = (float*)malloc(mem_size);
    float *h_tdata = (float*)malloc(mem_size);
    float *gold    = (float*)malloc(mem_size);

    float *d_idata, *d_cdata, *d_tdata;
    cudaMalloc(&d_idata, mem_size);
    cudaMalloc(&d_cdata, mem_size);
    cudaMalloc(&d_tdata, mem_size);

    // check parameters and calculate execution configuration
    if (nx % TILE_DIM || ny % TILE_DIM) {
        printf("nx and ny must be a multiple of TILE_DIM\n");
        goto error_exit;
    }

    if (TILE_DIM % BLOCK_ROWS) {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        goto error_exit;
        }

    // host
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          h_idata[j*nx + i] = j*nx + i;

    // correct result for error checking
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          gold[j*nx + i] = h_idata[i*nx + j];

    // device
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;

    // ------------
    // time kernels
    // ------------
    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

    // --------------
    // transposeNaive
    // --------------
    printf("%25s", "naive transpose");
    cudaMemset(d_tdata, 0, mem_size);
    // warmup
    transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(startEvent, 0);
    transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    postprocess(gold, h_tdata, nx * ny, ms);

    // ------------------
    // transposeCoalesced
    // ------------------
    printf("%25s", "coalesced transpose");
    cudaMemset(d_tdata, 0, mem_size);
    // warmup
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(startEvent, 0);
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    postprocess(gold, h_tdata, nx * ny, ms);

    // ------------------------
    // transposeNoBankConflicts
    // ------------------------
    printf("%25s", "conflict-free transpose");
    cudaMemset(d_tdata, 0, mem_size);
    // warmup
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(startEvent, 0);
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);
    postprocess(gold, h_tdata, nx * ny, ms);

    error_exit:
    // cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_tdata);
    cudaFree(d_cdata);
    cudaFree(d_idata);
    free(h_idata);
    free(h_tdata);
    free(h_cdata);
    free(gold);
}
