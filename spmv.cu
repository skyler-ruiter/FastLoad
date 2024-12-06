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
#include <papi.h>

// todo: make function to generate random matrices
// todo: make profiling sections for naive and fastLoad

// test if output is correct
void test_output(double *y, double *y_golden, int M) {
    int error_count = 0;
    double epsilon = 1e-6;
    for (int i = 0; i < M; i++) {
        if (fabs(y_golden[i] - y[i]) > epsilon) {
            error_count++;
            printf("error at %d: %f != %f\n", i, y_golden[i], y[i]);
        }
    }

    printf("error count: %d\n", error_count);
}

// cpu implementation of csc_spmv
void csc_spmv_cpu(int m, int n, int nnz, int *csc_ptr, int *csc_rowidx, double *csc_val, double *x, double *y) {
    for (int i = 0; i < m; i++) {
        for (int j = csc_ptr[i]; j < csc_ptr[i + 1]; j++) {
            y[i] += csc_val[j] * x[csc_rowidx[j]];
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

    cudaMalloc((void **)&d_csc_rowidx, nnz * sizeof(int));
    cudaMalloc((void **)&d_csc_ptr, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_csc_val, nnz * sizeof(double));

    cudaMalloc((void **)&d_x, n * sizeof(double));
    cudaMalloc((void **)&d_y, m * sizeof(double));

    cudaMemcpy(d_csc_ptr, csc_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_rowidx, csc_rowidx, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_val, csc_val, (nnz) * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_x, x, (n) * sizeof(double), cudaMemcpyHostToDevice); 

    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    // todo: add papi profiling

    for (int i=0; i<1000; i++)
    {      
      cudaMemset(d_y, 0, m * sizeof(double));
      CSC_SpMV_naive<<<32 * numSMs, 64>>>(m, n, nnz, d_csc_ptr, d_csc_rowidx, d_csc_val, d_x, d_y);
      cudaDeviceSynchronize();
    }

    // todo: add papi profiling stop

    cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_csc_ptr);
    cudaFree(d_csc_rowidx);
    cudaFree(d_csc_val);
    cudaFree(d_x);
    cudaFree(d_y);
}

void init(double* v, long const N) {
    for (long i = 0; i < N; ++i) {
        v[i] = 1.f/(rand() % 1024);
    }
}

void sparsify(double* v, long const N, double sparsity) {
    for (long i = 0; i < N; ++i) {
        if (rand() % 100 < sparsity) {
            v[i] = 0.f;
        }
    }
}

//-------------------------------------------------------//
//-------------------------------------------------------//
//-------------------------------------------------------//
int main(int argc, char ** argv) {
  
  assert(argc == 3);
  
  PAPI_library_init(PAPI_VER_CURRENT);

  long const t0 = PAPI_get_real_usec();

  long const M = atoi(argv[1]);
  long const N = atoi(argv[2]);

  printf("M: %ld, N: %ld\n", M, N);

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  double *test_A = (double*)calloc(9 * 9, sizeof(double));
  double *test_x = (double*)calloc(9, sizeof(double));
  double *test_y = (double*)calloc(9, sizeof(double));

  // Populate test_A with the test matrix
  double temp_A[9][9] = {
      {5, 0, 0, 0, 0, 0, 1, 0, 2},
      {3, 4, 5, 0, 0, 0, 6, 0, 7},
      {0, 8, 9, 0, 0, 0, 5, 0, 1},
      {2, 0, 3, 4, 5, 6, 0, 0, 0},
      {7, 0, 8, 9, 5, 1, 0, 0, 0},
      {2, 0, 3, 4, 5, 6, 0, 0, 0},
      {7, 0, 8, 9, 5, 1, 0, 0, 0},
      {2, 3, 4, 5, 6, 7, 8, 9, 5},
      {1, 2, 3, 4, 5, 6, 7, 8, 9}
  };

  for (int i = 0; i < 9; i++) {
      for (int j = 0; j < 9; j++) {
          test_A[i * 9 + j] = temp_A[i][j];
      }
  }

  for (int i = 0; i < 9; i++) {
      test_x[i] = 1;
  }

  // Convert test_A to CSC format
  int *test_csc_ptr = (int *)malloc((9 + 1) * sizeof(int));
  int *test_csc_rowidx = (int *)malloc(27 * sizeof(int));
  double *test_csc_val = (double *)malloc(27 * sizeof(double));

  int test_nnz = 0;
  for (int i = 0; i < 9; i++) {
      test_csc_ptr[i] = test_nnz;
      for (int j = 0; j < 9; j++) {
          if (test_A[j * 9 + i] != 0) {
              test_csc_rowidx[test_nnz] = j;
              test_csc_val[test_nnz] = test_A[j * 9 + i];
              test_nnz++;
          }
      }
  }
  test_csc_ptr[9] = test_nnz;

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Allocate host space for A, x, and y.
  double *hA = (double*)calloc(M * N, sizeof(double));
  double *hx = (double*)calloc(N, sizeof(double));
  double *hy = (double*)calloc(M, sizeof(double));

  // Initialize A and x.
  init(hA, M * N);
  init(hx, N);

  // sparse A
  sparsify(hA, M * N, 50);

  // count non-zero elements
  int nnz = 0;
  for (long i = 0; i < M * N; ++i) {
      if (hA[i] != 0) {
          nnz++;
      }
  }

  // Allocate space for CSC format
  int *csc_ptr = (int *)malloc((N + 1) * sizeof(int));
  int *csc_rowidx = (int *)malloc(nnz * sizeof(int));
  double *csc_val = (double *)malloc(nnz * sizeof(double));

  // Convert to CSC format
  int index = 0;
  for (int i = 0; i < N; i++) {
      csc_ptr[i] = index;
      for (int j = 0; j < M; j++) {
          if (hA[j * N + i] != 0) {
              csc_rowidx[index] = j;
              csc_val[index] = hA[j * N + i];
              index++;
          }
      }
  }
  csc_ptr[N] = index;

  // // check result is correct
  // double *y_golden = (double *)malloc(M * sizeof(double));
  // memset(y_golden, 0, M * sizeof(double));
  // for (int i = 0; i < M; i++) {
  //     for (int j = csc_ptr[i]; j < csc_ptr[i + 1]; j++) {
  //         y_golden[i] += csc_val[j] * hx[csc_rowidx[j]];
  //     }
  // }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // run regular csc_spmv
  //csc_spmv(M, N, nnz, csc_ptr, csc_rowidx, csc_val, hx, hy);

  // run csc_spmv with test matrix
  csc_spmv(9, 9, 27, test_csc_ptr, test_csc_rowidx, test_csc_val, test_x, test_y);

  // run test_a through cpu implementation
  double *y_golden = (double *)malloc(9 * sizeof(double));
  memset(y_golden, 0, 9 * sizeof(double));
  for (int i = 0; i < 9; i++) {
      for (int j = test_csc_ptr[i]; j < test_csc_ptr[i + 1]; j++) {
          y_golden[i] += test_csc_val[j] * test_x[test_csc_rowidx[j]];
      }
  }

  // check result is correct
  test_output(test_y, y_golden, 9);


  // int error_count = 0;
  // double epsilon = 1e-6;
  // for (int i = 0; i < M; i++) {
  //     if (fabs(y_golden[i] - hy[i]) > epsilon) {
  //         error_count++;
  //         printf("error at %d: %f != %f\n", i, y_golden[i], hy[i]);
  //     }
  // }

  // printf("error count: %d\n", error_count);
  
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  //                   FastLoad SpMV                      //
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//



  return 0;
}

