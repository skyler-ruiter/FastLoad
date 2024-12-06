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

// convert to CSC format from dense
void dense2csc(double* A, int M, int N, double* csc_val, int* csc_rowidx, int* csc_ptr) {
    int nnz = 0;
    for (int i = 0; i < N; i++) {
        csc_ptr[i] = nnz;
        for (int j = 0; j < M; j++) {
            if (A[j * N + i] != 0) {
                csc_val[nnz] = A[j * N + i];
                csc_rowidx[nnz] = j;
                nnz++;
            }
        }
    }
    csc_ptr[N] = nnz;
}

//*WORKS
// CPU implementation of csc_spmv
void csc_spmv_cpu(int m, int n, int nnz, int *csc_ptr, int *csc_rowidx, double *csc_val, double *x, double *y) {
  
  memset(y, 0, m * sizeof(double));
  
  for (int i = 0; i < n; i++) {
      for (int j = csc_ptr[i]; j < csc_ptr[i + 1]; j++) {
          int rowidx = csc_rowidx[j];
          double val = csc_val[j];
          y[rowidx] += val * x[i];
      }
  }
}

void generate_test_matrix(double* csc_val, int* csc_rowidx, int* csc_ptr) {
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

  double *test_A = (double*)calloc(9 * 9, sizeof(double));

  for (int i = 0; i < 9; i++) {
      for (int j = 0; j < 9; j++) {
          test_A[i * 9 + j] = temp_A[i][j];
      }
  }

  // // print A
  // std::cout << "A: " << std::endl;
  // for (int i = 0; i < 9; i++) {
  //     for (int j = 0; j < 9; j++) {
  //         std::cout << test_A[i * 9 + j] << " ";
  //     }
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;

  //convert to CSC
  dense2csc(test_A, 9, 9, csc_val, csc_rowidx, csc_ptr);

  // // print out csc format
  // std::cout << "CSC Vals: " << std::endl;
  // for (int i = 0; i < 50; i++) {
  //     std::cout << csc_val[i] << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "CSC RowIdx: " << std::endl;
  // for (int i = 0; i < 50; i++) {
  //     std::cout << csc_rowidx[i] << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "CSC Ptr: " << std::endl;
  // for (int i = 0; i < 10; i++) {
  //     std::cout << csc_ptr[i] << " ";
  // }
  // std::cout << std::endl;
}

//* WORKS
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

// *WORKS
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

void fastload(slide_matrix *matrix, int nnz, int N, int M, double *x, double *y) {

  int tilenum = matrix->tilenum;
  int *tile_ptr = matrix->tile_ptr;

  int *tile_len = matrix->tile_len;
  int *tile_colidx = matrix->tile_colidx;
  int *tile_format = matrix->tile_format;
  int *sortrowidx = matrix->sortrowidx;
  double *sortval = matrix->sortval;

  int segsum = matrix->segsum;
  int *sortrowindex = matrix->sortrowindex;

  int *d_tile_ptr;
  int *d_tile_len;
  int *d_tile_colidx;
  int *d_format;
  int *d_sortrowidx;
  double *d_sortval;
  int *d_sortrowindex;

  cudaMalloc((void **)&d_tile_ptr, (tilenum + 1) * sizeof(int));
  cudaMalloc((void **)&d_tile_len, (tilenum)*sizeof(int));
  cudaMalloc((void **)&d_tile_colidx, (tilenum)*sizeof(int));
  cudaMalloc((void **)&d_format, (tilenum)*sizeof(int));
  cudaMalloc((void **)&d_sortrowidx,(nnz)*sizeof(int));
  cudaMalloc((void **)&d_sortval,(nnz)*sizeof(double));
  cudaMalloc((void **)&d_sortrowindex,(nnz)*sizeof(int));

  cudaMemcpy(d_tile_ptr, tile_ptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tile_len, tile_len, (tilenum) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tile_colidx, tile_colidx, (tilenum) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_format, tile_format, (tilenum) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sortrowidx, sortrowidx, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sortval, sortval, (nnz) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sortrowindex, sortrowindex, (nnz) * sizeof(int), cudaMemcpyHostToDevice);

  double *d_x;
  double *d_y;

  cudaMalloc((void **)&d_x, N * sizeof(double));
  cudaMalloc((void **)&d_y, M * sizeof(double));

  cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

  int num_threads = slidesize *warpperblock;
  int num_blocks = ceil(( double)tilenum / (double)warpperblock); 

  FastLoad___kernel<<<num_blocks, num_threads>>>(M,N,nnz,
                                                 tilenum,
                                                 d_tile_ptr,
                                                 d_tile_len,
                                                 d_tile_colidx,
                                                 d_format,
                                                 d_sortrowindex,
                                                 d_sortrowidx,
                                                 d_sortval,
                                                 d_x,
                                                 d_y);
  cudaDeviceSynchronize();
  
  cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_tile_ptr);
  cudaFree(d_tile_len);
  cudaFree(d_tile_colidx);
  cudaFree(d_format);
  cudaFree(d_sortrowidx);
  cudaFree(d_sortval);
  cudaFree(d_sortrowindex);
}

int main(int argc, char ** argv)
{
  assert(argc == 3);

  // printout the matrix size
  long const M = atoi(argv[1]);
  long const N = atoi(argv[2]);
  printf("M: %ld, N: %ld\n", M, N);

  { // correctness testing
    // // allocate memory for test csc matrix of 9x9 matrix 50 nnz
    // double *test_csc_val = (double *)malloc(50 * sizeof(double));
    // int *test_csc_rowidx = (int *)malloc(50 * sizeof(int));
    // int *test_csc_ptr = (int *)malloc(10 * sizeof(int));

    // // generate test matrix
    // generate_test_matrix(test_csc_val, test_csc_rowidx, test_csc_ptr);

    // // allocate memory for x and y
    // double *test_x = (double *)malloc(N * sizeof(double));
    // double *test_y = (double *)malloc(M * sizeof(double));

    // // initialize x to be 1-9
    // for (int i = 0; i < 9; i++) {
    //     test_x[i] = i + 1;
    // }

    // // call csc_spmv
    // csc_spmv(9, 9, 50, test_csc_ptr, test_csc_rowidx, test_csc_val, test_x, test_y);

    // // call csc_spmv_cpu
    // csc_spmv_cpu(9, 9, 50, test_csc_ptr, test_csc_rowidx, test_csc_val, test_x, test_y);
  }

  // allocate memory for matrices
  double *dense_A = (double *)malloc(M * N * sizeof(double));
  double *x = (double *)malloc(N * sizeof(double));
  double *y = (double *)malloc(M * sizeof(double));
  memset(y, 0, M * sizeof(double));

  // initialize dense_A and x
  init(dense_A, M * N);
  init(x, N);

  // sparsify dense_A
  sparsify(dense_A, M * N, 50);

  // print A and x
  // std::cout << "A: " << std::endl;
  // for (int i = 0; i < M; i++) {
  //     for (int j = 0; j < N; j++) {
  //         std::cout << dense_A[i * N + j] << " ";
  //     }
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;
  // std::cout << "x: " << std::endl;
  // for (int i = 0; i < N; i++) {
  //     std::cout << x[i] << " ";
  // }
  // std::cout << std::endl;

  // get nnz
  int nnz = 0;
  for (int i = 0; i < M * N; i++) {
      if (dense_A[i] != 0) {
          nnz++;
      }
  }

  // allocate memory for csc matrix
  double *csc_val = (double *)malloc(nnz * sizeof(double));
  int *csc_rowidx = (int *)malloc(nnz * sizeof(int));
  int *csc_ptr = (int *)malloc((N + 1) * sizeof(int));

  // convert dense_A to csc format
  dense2csc(dense_A, M, N, csc_val, csc_rowidx, csc_ptr);

  // if M=N=9 use the test matrix
  if (M == 9 && N == 9) {
    csc_val = (double *)malloc(50 * sizeof(double));
    csc_rowidx = (int *)malloc(50 * sizeof(int));
    csc_ptr = (int *)malloc(10 * sizeof(int));
    generate_test_matrix(csc_val, csc_rowidx, csc_ptr);
    nnz = 50;
    // make x 1-9
    for (int i = 0; i < 9; i++) {
        x[i] = i + 1;
    }
    std::cout << "Using test matrix" << std::endl;
  }

  // call csc_spmv
  csc_spmv(M, N, nnz, csc_ptr, csc_rowidx, csc_val, x, y);

  // print out y
//   std::cout << "y: " << std::endl;
//   for (int i = 0; i < M; i++) {
//       std::cout << y[i] << " ";
//   }
//   std::cout << std::endl;

  // golden y
  double *y_golden = (double *)malloc(M * sizeof(double)); 
  memset(y_golden, 0, M * sizeof(double));
  
  // call csc_spmv_cpu
  csc_spmv_cpu(M, N, nnz, csc_ptr, csc_rowidx, csc_val, x, y_golden);

  // print out y_golden
//   std::cout << "y_golden: " << std::endl;
//     for (int i = 0; i < M; i++) {
//         std::cout << y_golden[i] << " ";
//     }
//     std::cout << std::endl;


  // check result is correct
  int error_count_check = 0;
  float error_threshold = 0.0001;
  for (int i = 0; i < M; i++) {
      if (abs(y_golden[i] - y[i]) > error_threshold) {
          error_count_check++;
      }
  }

  // print out error count
  printf("Error count: %d\n", error_count_check);

//   // print csc format
//   std::cout << "CSC Vals: " << std::endl;
//     for (int i = 0; i < nnz; i++) {
//         std::cout << csc_val[i] << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "CSC RowIdx: " << std::endl;
//     for (int i = 0; i < nnz; i++) {
//         std::cout << csc_rowidx[i] << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "CSC Ptr: " << std::endl;
//     for (int i = 0; i < N + 1; i++) {
//         std::cout << csc_ptr[i] << " ";
//     }
//     std::cout << std::endl;

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // allocate space for fastload y
  double *y_fastload = (double *)malloc(M * sizeof(double));
  memset(y_fastload, 0, M * sizeof(double));

  int *nnzpercol = (int *)malloc(N * sizeof(int));
  memset(nnzpercol, 0, sizeof(int) * N);

  // get nnzpercol
  for (int i = 0; i < N; i++) {
      nnzpercol[i] = csc_ptr[i + 1] - csc_ptr[i];
  }

//   // print out nnzpercol
//   for (int i = 0; i < N; i++) {
//       printf("%d ", nnzpercol[i]);
//   }
//   printf("\n");

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

  int h_count;
  double timeFormatTran=0;
  double timeFortmatClas=0;

  slide_matrix *matrixA = (slide_matrix *)malloc(sizeof(slide_matrix));

  // on GPU
  formatTransform(timeFormatTran, timeFortmatClas, matrixA, sortrowidx_tmp, sortval_tmp, sortnnz_tmp, nnz, N, M, h_count);

    // print out sortnnz_tmp
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", sortnnz_tmp[i]);
    // }
    // printf("\n");

    // // print out sortval_tmp and sortrowidx_tmp
    // for (int i = 0; i < nnz; i++) {
    //     // convert to int if M=N=9
    //     if (M == 9 && N == 9) {
    //         printf("%d ", (int)sortval_tmp[i]);
    //     } else {
    //         printf("%f ", sortval_tmp[i]);
    //     }
    // }
    // printf("\n");
    // for (int i = 0; i < nnz; i++) {
    //     printf("%d ", sortrowidx_tmp[i]);
    // }
    // printf("\n");

    // // print sorted X
    // for (int i = 0; i < N; i++) {
    //     printf("%f ", sortx[i]);
    // }
    // printf("\n");

  free(nnzpercol);
  free(sortrowidx_tmp);
  free(sortval_tmp);
  free(sortnnz_tmp);

  // call fastload
  //fastload(matrixA, nnz, M, N, x, y_fastload);
  char *filename = "9x9.mtx";
  FastLoad_spmv(filename, matrixA, nnz, M, N, sortx, y_fastload, y_golden);

    // print out y_fastload
    // std::cout << "y_fastload: " << std::endl;
    // for (int i = 0; i < M; i++) {
    //     std::cout << y_fastload[i] << " ";
    // }
    // std::cout << std::endl;

  // check result is correct
  int error_count_check_fastload = 0; 
  for (int i = 0; i < M; i++) {
      if (abs(y_golden[i] - y_fastload[i]) > error_threshold) {
          error_count_check_fastload++;
      }
  }

  // print out error count
  printf("Error count fastload: %d\n", error_count_check_fastload);

  // free memory
    free(dense_A);
    free(x);
    free(y);
    free(csc_val);
    free(csc_rowidx);
    free(csc_ptr);
    free(y_golden);
    free(y_fastload);

  return 0;
}