#include "src/coo2csc.h"
#include "sys/time.h"
#include "src/FastLoad_CPU.h"
#include "src/FastLoad_GPU.h"
#include "src/CSCSpMV.h"
#include "src/csr2csc_cuda.h"
#include "src/formatTransform_GPU.h"
#include "src/ColSort_GPU.h"

#include "src/cusparse_cuda.h"
#include <iostream>

void init(double* v, long const N) {
    #pragma omp parallel for
    for (long i = 0; i < N; ++i) {
        v[i] = 1.f/(rand() % 1024);
    }
}

void sparsify(double* v, long const N, int sparsity) {
    #pragma omp parallel for
    for (long i = 0; i < N; ++i) {
        if ((rand() % 100) + 1 < sparsity) {
            v[i] = 0.f;
        }
    }
}

// method to generate a random csc sparse matrix
void generate_random_csc_matrix(int M, int N, double sparsity, double* csc_val, int* csc_rowidx, int* csc_ptr) {
    int nnz = 0;
    for (int i = 0; i < N; i++) {
        csc_ptr[i] = nnz;
        for (int j = 0; j < M; j++) {
            if ((rand() % 100) > sparsity) {
                csc_val[nnz] = static_cast<double>(rand()) / RAND_MAX;
                csc_rowidx[nnz] = j;
                nnz++;
            }
        }
    }
    csc_ptr[N] = nnz;
}

// CPU implementation of csc_spmv
void csc_spmv_cpu(int m, int n, int nnz, int *csc_ptr, int *csc_rowidx, double *csc_val, double *x, double *y) {
  memset(y, 0, m * sizeof(double));
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
      for (int j = csc_ptr[i]; j < csc_ptr[i + 1]; j++) {
          int rowidx = csc_rowidx[j];
          double val = csc_val[j];
          #pragma omp atomic
          y[rowidx] += val * x[i];
      }
  }
}


__global__ void CSC_SpMV_naive(int m, int n, int nnz, int *csc_ptr, int *csc_rowIdx, double *csc_val, double *x, double *y) 
{
    int global_id = blockIdx.x *blockDim.x + threadIdx.x;
    for (int i = global_id; i < n; i += blockDim.x * gridDim.x) 
    {
        double dotProduct = 0;
        const int col_start = csc_ptr[i];
        const int col_end = csc_ptr[i + 1];
        
        for (int j = col_start; j < col_end; j++) 
        {
            dotProduct = csc_val[j] * x[i];
            int rowidx_tmp = csc_rowIdx[j];
            atomicAdd(&y[rowidx_tmp], dotProduct);
        }  
    }
}

void csc_spmv(int m, int n, int nnz, int *csc_ptr, int *csc_rowidx, double *csc_val, double *x, double *y)
{
    int *d_csc_rowidx;
    int *d_csc_ptr;
    double *d_csc_val;
    double *d_x;
    double *d_y;
    int numSMs;
    int numTests = 100;

    cudaMalloc((void **)&d_csc_rowidx, nnz * sizeof(int));
    cudaMalloc((void **)&d_csc_ptr, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_csc_val, nnz * sizeof(double));
    cudaMalloc((void **)&d_x, n * sizeof(double));
    cudaMalloc((void **)&d_y, m * sizeof(double));

    timeval t1, t2;
    double time_memcpy_h2d = 0, time_memcpy_d2h = 0, time_kernel = 0;

    gettimeofday(&t1, NULL);
    cudaMemcpy(d_csc_ptr, csc_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_rowidx, csc_rowidx, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_val, csc_val, (nnz) * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_x, x, (n) * sizeof(double), cudaMemcpyHostToDevice); 
    gettimeofday(&t2, NULL);
    
    time_memcpy_h2d = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    // dry run
    CSC_SpMV_naive<<<32 * numSMs, 64>>>(m, n, nnz, d_csc_ptr, d_csc_rowidx, d_csc_val, d_x, d_y);
    cudaMemset(d_y, 0, m * sizeof(double));
    cudaDeviceSynchronize();

    // make array to store the times
    double *times = (double *)malloc(numTests * sizeof(double));

    for (int i=0; i<numTests; i++) {      
      cudaMemset(d_y, 0, m * sizeof(double));
      gettimeofday(&t1, NULL);
      CSC_SpMV_naive<<<32 * numSMs, 64>>>(m, n, nnz, d_csc_ptr, d_csc_rowidx, d_csc_val, d_x, d_y);
      cudaDeviceSynchronize();
      gettimeofday(&t2, NULL);
      times[i] = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }
    
    // find average kernel time
    for (int i=0; i<numTests; i++) {
      time_kernel += times[i];
    }
    time_kernel /= numTests;

    gettimeofday(&t1, NULL);
    cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost);
    gettimeofday(&t2, NULL);
    time_memcpy_d2h = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    double gflops = 2 * (double)nnz * 1.0e-6 / time_kernel;

    printf("CSC SpMV Time: %f ms\n", time_memcpy_h2d + time_kernel + time_memcpy_d2h);
    printf("Memory H2D Time: %f ms\n", time_memcpy_h2d);
    printf("Kernel Time: %f ms\n", time_kernel);
    printf("Memory D2H Time: %f ms\n", time_memcpy_d2h);
    printf("GFLOPS: %f\n", gflops);

    cudaFree(d_csc_ptr);
    cudaFree(d_csc_rowidx);
    cudaFree(d_csc_val);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char ** argv)
{
  assert(argc == 4);

  // printout the matrix size
  long const M = atoi(argv[1]);
  long const N = atoi(argv[2]);
  printf("M: %ld, N: %ld\n", M, N);

  // get sparsity
  int const sparsity = atof(argv[3]);

  // initialize x and y
  double *x = (double *)malloc(N * sizeof(double));
  double *y = (double *)malloc(M * sizeof(double));
  memset(y, 0, M * sizeof(double));

  // initialize x
  init(x, N);

  int nnz = 0;

  // allocate memory for csc matrix
  double *csc_val = (double *)malloc(M * N * sizeof(double)); // over-allocate to handle sparsity
  int *csc_rowidx = (int *)malloc(M * N * sizeof(int)); // over-allocate to handle sparsity
  int *csc_ptr = (int *)malloc((N + 1) * sizeof(int));

  // generate random csc matrix
  generate_random_csc_matrix(M, N, sparsity, csc_val, csc_rowidx, csc_ptr);

  // get nnz
  nnz = csc_ptr[N];
  printf("nnz: %d\n\n", nnz);

  // call csc_spmv
  csc_spmv(M, N, nnz, csc_ptr, csc_rowidx, csc_val, x, y);

  // golden y
  double *y_golden = (double *)malloc(M * sizeof(double)); 
  memset(y_golden, 0, M * sizeof(double));
  
  // call csc_spmv_cpu
  csc_spmv_cpu(M, N, nnz, csc_ptr, csc_rowidx, csc_val, x, y_golden);

  // check result is correct
  int error_count_check = 0;
  float error_threshold = 0.0001;
  for (int i = 0; i < M; i++) {
      if (abs(y_golden[i] - y[i]) > error_threshold) {
          error_count_check++;
      }
  }

  if (error_count_check != 0) {
      printf("Error: %d\n", error_count_check);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // FASTLOAD SECTION

  // allocate space for fastload y
  double *y_fastload = (double *)malloc(M * sizeof(double));
  memset(y_fastload, 0, M * sizeof(double));

  int *nnzpercol = (int *)malloc(N * sizeof(int));
  memset(nnzpercol, 0, sizeof(int) * N);

  // get nnzpercol
  for (int i = 0; i < N; i++) {
      nnzpercol[i] = csc_ptr[i + 1] - csc_ptr[i];
  }

  int *sortrowidx_tmp = (int *)malloc(sizeof(int)*nnz);
  memset(sortrowidx_tmp,0,sizeof(int)*nnz);
  double *sortval_tmp = (double *)malloc(sizeof(double)*nnz);
  memset(sortval_tmp,0,sizeof(double)*nnz);
  int *sortnnz_tmp= (int *)malloc(sizeof(int)*(N));
  memset(sortnnz_tmp,0,sizeof(int)*N);
  double *sortx = (double *)malloc(sizeof(double)*N);  
  memset(sortx,0,sizeof(double)*N);

  // sort columns
  double timeForSort = 0;
  // on GPU
  ColSort(timeForSort, M, N, nnz, nnzpercol, csc_ptr, csc_rowidx, csc_val, sortrowidx_tmp, sortval_tmp, sortnnz_tmp, x, sortx);

  printf("\nFastLoad Times:\n");

  // print time for sort
  printf("Time for sort: %f ms\n", timeForSort);

  int h_count;
  double timeFormatTran=0;
  double timeFortmatClas=0;

  slide_matrix *matrixA = (slide_matrix *)malloc(sizeof(slide_matrix));

  // on GPU
  formatTransform(timeFormatTran, timeFortmatClas, matrixA, sortrowidx_tmp, sortval_tmp, sortnnz_tmp, nnz, N, M, h_count);

  // print time for format transform
  printf("Time for format transform: %f ms\n", timeFormatTran);
  printf("Time for format classification: %f ms\n", timeFortmatClas);

  // print total pre-processing time
  printf("Total pre-processing time: %f ms\n", timeForSort + timeFormatTran + timeFortmatClas);

  free(nnzpercol);
  free(sortrowidx_tmp);
  free(sortval_tmp);
  free(sortnnz_tmp);

  // call fastload
  char *filename = "temp.mtx";
  FastLoad_spmv(filename, matrixA, nnz, M, N, sortx, y_fastload, y_golden);

  // free memory
  free(x);
  free(y);
  free(csc_val);
  free(csc_rowidx);
  free(csc_ptr);
  free(y_golden);
  free(y_fastload);

  return 0;
}