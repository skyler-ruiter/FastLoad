#include "src/FastLoad_CPU.h"
#include "src/FastLoad_GPU.h"
#include "src/CSCSpMV.h"
#include <iostream>
#include <chrono>

// method to generate a random csc sparse matrix
void generate_random_csc_matrix(int M, int N, double sparsity, double* csc_val, int* csc_rowidx, int* csc_ptr) {
    int nnz = 0;
    for (int i = 0; i < N; i++) {
        csc_ptr[i] = nnz;
        for (int j = 0; j < M; j++) {
            if ((rand() % 100) < sparsity) {
                csc_val[nnz] = static_cast<double>(rand()) / RAND_MAX;
                csc_rowidx[nnz] = j;
                nnz++;
            }
        }
    }
    csc_ptr[N] = nnz;
}

void benchmark_csc_spmv(int M, int N, int nnz, int *csc_ptr, int *csc_rowidx, double *csc_val, double *x, double *y) {
    auto start = std::chrono::high_resolution_clock::now();
    csc_spmv(M, N, nnz, csc_ptr, csc_rowidx, csc_val, x, y);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    double gflops = (2.0 * nnz) / (time * 1e9);
    double bandwidth = (nnz * (sizeof(int) + sizeof(double)) + N * sizeof(double) + M * sizeof(double)) / (time * 1e9);
    std::cout << "CSC SpMV Time: " << time << " s\n";
    std::cout << "CSC SpMV GFLOPS: " << gflops << "\n";
    std::cout << "CSC SpMV Bandwidth: " << bandwidth << " GB/s\n";
}

void benchmark_fastload_spmv(slide_matrix *matrix, int nnz, int N, int M, double *y, double *sortx, double *y_fastload, double *y_golden) {
    char *filename = "9x9.mtx";
    auto start = std::chrono::high_resolution_clock::now();
    FastLoad_spmv(filename, matrix, nnz, N, M, sortx, y_fastload, y_golden);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    double gflops = (2.0 * nnz) / (time * 1e9);
    double bandwidth = (nnz * (sizeof(int) + sizeof(double)) + N * sizeof(double) + M * sizeof(double)) / (time * 1e9);
    std::cout << "FastLoad SpMV Time: " << time << " s\n";
    std::cout << "FastLoad SpMV GFLOPS: " << gflops << "\n";
    std::cout << "FastLoad SpMV Bandwidth: " << bandwidth << " GB/s\n";
}

int main(int argc, char **argv) {
    assert(argc == 4);
    long const M = atoi(argv[1]);
    long const N = atoi(argv[2]);
    printf("M: %ld, N: %ld\n", M, N);
    long const nnz = atoi(argv[3]);
    double sparsity;
    if (nnz == 0) {
        sparsity = 10;
    } else {
        sparsity = (double)nnz / (M * N) * 100;
    }

    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(M * sizeof(double));
    memset(y, 0, M * sizeof(double));
    init(x, N);

    double *csc_val = (double *)malloc(M * N * sizeof(double));
    int *csc_rowidx = (int *)malloc(M * N * sizeof(int));
    int *csc_ptr = (int *)malloc((N + 1) * sizeof(int));
    generate_random_csc_matrix(M, N, sparsity, csc_val, csc_rowidx, csc_ptr);
    int nnz = csc_ptr[N];

    double *y_golden = (double *)malloc(M * sizeof(double)); 
    memset(y_golden, 0, M * sizeof(double));
  
    // call csc_spmv_cpu
    csc_spmv_cpu(M, N, nnz, csc_ptr, csc_rowidx, csc_val, x, y_golden);

    benchmark_csc_spmv(M, N, nnz, csc_ptr, csc_rowidx, csc_val, x, y);

    double *y_fastload = (double *)malloc(M * sizeof(double));
    memset(y_fastload, 0, M * sizeof(double));

    int *nnzpercol = (int *)malloc(N * sizeof(int));
    memset(nnzpercol, 0, sizeof(int) * N);
    for (int i = 0; i < N; i++) {
        nnzpercol[i] = csc_ptr[i + 1] - csc_ptr[i];
    }

    int *sortrowidx_tmp = (int *)malloc(sizeof(int) * nnz);
    memset(sortrowidx_tmp, 0, sizeof(int) * nnz);
    double *sortval_tmp = (double *)malloc(sizeof(double) * nnz);
    memset(sortval_tmp, 0, sizeof(double) * nnz);
    int *sortnnz_tmp = (int *)malloc(sizeof(int) * N);
    memset(sortnnz_tmp, 0, sizeof(int) * N);
    double *sortx = (double *)malloc(sizeof(double) * N);
    memset(sortx, 0, sizeof(double) * N);

    double timeForSort = 0;
    ColSort(timeForSort, M, N, nnz, nnzpercol, csc_ptr, csc_rowidx, csc_val, sortrowidx_tmp, sortval_tmp, sortnnz_tmp, x, sortx);

    int h_count;
    double timeFormatTran = 0;
    double timeFortmatClas = 0;
    slide_matrix *matrixA = (slide_matrix *)malloc(sizeof(slide_matrix));
    formatTransform(timeFormatTran, timeFortmatClas, matrixA, sortrowidx_tmp, sortval_tmp, sortnnz_tmp, nnz, N, M, h_count);

    free(nnzpercol);
    free(sortrowidx_tmp);
    free(sortval_tmp);
    free(sortnnz_tmp);

    benchmark_fastload_spmv(matrixA, nnz, N, M, y, sortx, y_fastload, y_golden);

    free(x);
    free(y);
    free(csc_val);
    free(csc_rowidx);
    free(csc_ptr);
    free(y_fastload);

    return 0;
}
