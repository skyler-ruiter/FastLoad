#include <cuda_runtime_api.h> 
#include <cusparse.h>         
//#include <stdio.h>            
//#include <stdlib.h>          


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        /*return EXIT_FAILURE;  */                                                 \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        /*return EXIT_FAILURE; */                                                \
    }                                                                          \
}

void call_cusparse_spmv(char *filename,
                        const int A_nnz,
                        const int A_num_rows,
                        const int A_num_cols,
                        int *hA_csrOffsets,
                        int *hA_columns,
                        double *hA_values,
                        double *hX,
                        double *hY,
                        double *hY_result)

{
    // Host problem definition

    double     alpha           = 1.0;
    double     beta            = 0.0;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    double *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(double))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(double),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_64F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_64F) )
    // allocate an external buffer if needed


    timeval tcusparse1, tcusparse2;
    double time_cuda_spmv_cusparse = 0;
    gettimeofday(&tcusparse1, NULL);

    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

    cudaDeviceSynchronize();
    gettimeofday(&tcusparse2, NULL);

    time_cuda_spmv_cusparse = (tcusparse2.tv_sec - tcusparse1.tv_sec) * 1000.0 + (tcusparse2.tv_usec - tcusparse1.tv_usec) / 1000.0;
    double gflops_cusparse = 2 * (double)A_nnz * 1.0e-6 / time_cuda_spmv_cusparse;
    //printf("time_cusparse %f cusparse_gflops %f \n", time_cuda_spmv_cusparse, gflops_cusparse) ;

    double compression_rate = (double)A_nnz / (double)A_num_rows;


   
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(double),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (hY[i] != hY_result[i]) { // direct floating point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
    {
        printf("cuSPARSE test PASSED! (%f ms %4.2f GFlops)\n", time_cuda_spmv_cusparse, gflops_cusparse);
        FILE *fout = fopen("TimeResult/results_cusparse.txt", "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "%s rowA %i colA %i nnz %i cusparse %f Gflops %f \n",
                filename, A_num_rows, A_num_cols, A_nnz, time_cuda_spmv_cusparse, gflops_cusparse);
       fclose(fout);
    }
    else
    {
        printf("cuSPARSE test FAILED: wrong result\n");
        FILE *fout = fopen("TimeResult/results_cusparse.txt", "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "error mtx: %s  \n",
                        filename);
       fclose(fout);
    }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    //return EXIT_SUCCESS;
}
