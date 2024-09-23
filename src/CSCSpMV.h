#include <cuda_runtime.h>
#include <iostream>

__global__ void CSC_SpMV_kernel(int m,
                                int n,
                                int nnz,
                                int *csc_ptr,
                                int *csc_rowidx,
                                double *csc_val,
                                double *x,
                                double *y)
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
            int rowidx_tmp = csc_rowidx[j];
            atomicAdd(&y[rowidx_tmp], dotProduct);
        }  
    }
}

void csc_spmv(char *filename,
              int m,
              int n,
              int nnz,
              int *csc_ptr,
              int *csc_rowidx,
              double *csc_val,
              
              double *x,
              double *y,
              double *y_golden)
{
    int *d_csc_rowidx;
    int *d_csc_ptr;
    double *d_csc_val;

    double *d_x;
    double *d_y;

    int numSMs;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_csc_rowidx, nnz * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_csc_ptr, (n+1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_csc_val, nnz * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_x, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_y, m * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_csc_ptr, csc_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csc_rowidx, csc_rowidx, (nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csc_val, csc_val, (nnz) * sizeof(double), cudaMemcpyHostToDevice)); 
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, (n) * sizeof(double), cudaMemcpyHostToDevice)); 


    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    timeval tcsc1, tcsc2;
    double time_cuda_csc_spmv_base = 0;

    for(int i=0; i<1000; i++)
    {      
        cudaMemset(d_y, 0, m * sizeof(double));
        gettimeofday(&tcsc1, NULL);  
        CSC_SpMV_kernel<<<32 * numSMs, 64>>>(m,
                                             n,
                                             nnz,
                                             d_csc_ptr,
                                             d_csc_rowidx,
                                             d_csc_val,
                                             d_x,
                                             d_y);
        cudaDeviceSynchronize();
        gettimeofday(&tcsc2, NULL);
        time_cuda_csc_spmv_base += (tcsc2.tv_sec - tcsc1.tv_sec) * 1000.0 + (tcsc2.tv_usec - tcsc1.tv_usec) / 1000.0;
    }
    time_cuda_csc_spmv_base /= 1000;
    double gflops = 2 * (double)nnz * 1.0e-6 / time_cuda_csc_spmv_base;
    //printf("CUDA_CSC_BASIC_SpMV_runtime %f ms , %4.2f GFlops\n", time_cuda_csc_spmv_base, gflops);

    
    CHECK_CUDA_ERROR(cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost));

    int error_count_basic_csc_d = 0;
    for (int i = 0; i < m; i++)
        if (abs(y_golden[i] - y[i]) > 0.01 * abs(y[i]))
        {
            error_count_basic_csc_d++;
        }

    if (error_count_basic_csc_d == 0)
    {
        printf("BASIC CSC GPU PASS! (%f ms %4.2f GFlops)\n",time_cuda_csc_spmv_base, gflops);
        FILE *fout = fopen("TimeResult/results_CSCSpMV.txt", "a");
        if (fout == NULL)
        printf("Writing results fails.\n");
        fprintf(fout,"%s m %d n %d nnz %d BasicCSC %f GFlops %f \n",
                filename, m,n,nnz,time_cuda_csc_spmv_base, gflops);
        fclose(fout);
    }
    else
    {
        printf("BASIC CSC check NO PASS! error_count_cuda = %d \n", error_count_basic_csc_d);
        FILE *fout = fopen("TimeResult/results_CSCSpMV.txt", "a");
        if (fout == NULL)
        printf("Writing results fails.\n");
        fprintf(fout,"error mtx %s \n",
                      filename);
        fclose(fout);
    }


    CHECK_CUDA_ERROR(cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost));


    cudaFree(d_csc_ptr);
    cudaFree(d_csc_rowidx);
    cudaFree(d_csc_val);
    cudaFree(d_x);
    cudaFree(d_y);




}