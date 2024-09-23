#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA slide Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA slide Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

__global__ void checkIndex(void)
{
    int global_id = blockIdx.x *blockDim.x + threadIdx.x;
    const int th_tile = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (slidesize -1) & threadIdx.x;
    printf("globalid: %d, th_tile: %d, local_warp_id: %d, lane_id: %d \n", global_id, th_tile,local_warp_id,lane_id);

}

__inline__ __device__ int warpReduce(int localSum)
{
    localSum += __shfl_xor(localSum, 16);
    localSum += __shfl_xor(localSum, 8);
    localSum += __shfl_xor(localSum, 4);
    localSum += __shfl_xor(localSum, 2);
    localSum += __shfl_xor(localSum, 1);
    return localSum;
}

__device__ __forceinline__
double __shfl_down1(double var, unsigned int srcLane, int width=32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down1(a.x, srcLane, width);
    a.y = __shfl_down1(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

enum
{
    /// The number of warp scan steps
    STEPS = 5,

    // The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    SHFL_C = ((-1 << STEPS) & 31) << 8
    //SHFL_C = 0
};



__forceinline__ __device__
double scan_32_shfl(double    x)
{
    #pragma unroll
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
        // Use predicate set from SHFL to guard against invalid peers
        asm(
            "{"
            "  .reg .s32 lo;"
            "  .reg .s32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 %0, {lo, hi};"
            "  @p add.f64 %0, %0, %1;"
            "}"
            : "=d"(x) : "d"(x), "r"(1 << STEP), "r"(SHFL_C));
    }
    return x;
}




__device__ __forceinline__ 
double ScanWarp(double val) {
  double result;
  asm("{"
      ".reg .f64 r<5>;"
      ".reg .pred p<5>;"

      "shfl.sync.up.b32 r0|p0, %1, 1, 0, -1;"
      "@p0 add.f64 r0, r0, %1;"

      "shfl.sync.up.b32 r1|p1, r0, 2, 0, -1;"
      "@p1 add.f64 r1, r1, r0;"

      "shfl.sync.up.b32 r2|p2, r1, 4, 0, -1;"
      "@p2 add.f64 r2, r2, r1;"

      "shfl.sync.up.b32 r3|p3, r2, 8, 0, -1;"
      "@p3 add.f64 r3, r3, r2;"

      "shfl.sync.up.b32 r4|p4, r3, 16, 0, -1;"
      "@p4 add.f64 r4, r4, r3;"

      "mov.f64 %0, r4;"
      "}"
      : "=d"(result)
      : "d"(val));
  return result;
}

__device__ double ScanWarp1(double val, const int localid) {
  //double lane = threadIdx.x & 31;
  double tmp = __shfl_up_sync(0xffffffff, val, 1);
  if (localid >= 1) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 2);
  if (localid >= 2) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 4);
  if (localid >= 4) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 8);
  if (localid >= 8) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 16);
  if (localid >= 16) {
    val += tmp;
  }
  return val;
}

__global__ void FastLoad___kernel(int rowA, int colA,int nnz,
                                  int tilenum,
                                  int *d_tile_ptr,
                                  //int *d_tile_tall,
                                  int *d_tile_len,
                                  int *d_tile_colidx,
                                  int *d_format,
                                  //int *d_countrow,
                                  //int *d_segmentoffset,
                                  int *d_sortrowindex,
                                  int *d_sortrowidx,
                                  double *d_sortval,
                                  double *d_x,
                                  double *d_y)
{
    int global_id = blockIdx.x *blockDim.x + threadIdx.x;
    const int th_tile = global_id >> 5;
    __shared__ double s_x[warpperblock *slidesize];
    __shared__ double s_y[warpperblock *slidesize];
    __shared__ double s_s[warpperblock *slidesize];

    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (slidesize -1) & threadIdx.x;
    double *s_x_warp = &s_x[local_warp_id *slidesize];
    double *s_y_warp = &s_y[local_warp_id *slidesize];
    double *s_s_warp = &s_s[local_warp_id *slidesize];
    //double yval =0;

    if(th_tile < tilenum)
    {
        int tilelen = d_tile_len[th_tile];
        int tile_start = d_tile_ptr[th_tile];
        int tile_stop = d_tile_ptr[th_tile+1];
        //int tile_tall = d_tile_tall[th_tile];
        int tile_colidx = d_tile_colidx[th_tile];
        int format = d_format[th_tile];
        switch(format)
        {
            case 0:
            {
                int j = tile_start + lane_id ;
                int d_rowidx = d_sortrowidx[j];
                if(lane_id <tilelen)
                {
                    s_x_warp[lane_id] = d_x[tile_colidx+lane_id];
                    atomicAdd(&d_y[d_rowidx], d_sortval[j] * s_x_warp[lane_id]);
          
                }
            }
            break;
            
            case 1:
            {
                int j = tile_start + lane_id ;
                int d_rowidx = d_sortrowidx[j];
                double val=0;
                //s_y_warp[lane_id] = d_sortval[j] * d_x[tile_colidx+lane_id];
                val =  d_sortval[j] * d_x[tile_colidx + lane_id] ;


                double tmp_sum =__shfl_up_sync(0xffffffff,val,1);
                if(lane_id /1 !=0)
                {val += tmp_sum;}

                tmp_sum =__shfl_up_sync(0xffffffff,val,2);
                if(lane_id /2 !=0)
                {val += tmp_sum;}

                tmp_sum =__shfl_up_sync(0xffffffff,val,4);
                if(lane_id /4 !=0)
                {val += tmp_sum;}

                tmp_sum =__shfl_up_sync(0xffffffff,val,8);
                if(lane_id / 8 != 0)
                {val += tmp_sum;}

                tmp_sum =__shfl_up_sync(0xffffffff,val,16);
                if(lane_id /16 != 0)
                {val += tmp_sum;}
                

                 if(lane_id ==31)
                    {
                        atomicAdd(&d_y[d_rowidx], val);
                    }   
            }

            break;
            

            case 2:
            {
                int j = tile_start + lane_id ;
                int d_rowidx = d_sortrowidx[j];
                double val=0;
                double val_tmp=0;
                //s_y_warp[lane_id] = d_sortval[j] * d_x[tile_colidx+lane_id];
                val =  d_sortval[j] * d_x[tile_colidx + lane_id] ;
                val_tmp =  d_sortval[j] * d_x[tile_colidx + lane_id];

                double tmp_sum =__shfl_up_sync(0xffffffff,val,1);
                if(lane_id /1 !=0)
                {val += tmp_sum;}

                tmp_sum =__shfl_up_sync(0xffffffff,val,2);
                if(lane_id /2 !=0)
                {val += tmp_sum;}

                tmp_sum =__shfl_up_sync(0xffffffff,val,4);
                if(lane_id /4 !=0)
                {val += tmp_sum;}

                tmp_sum =__shfl_up_sync(0xffffffff,val,8);
                if(lane_id / 8 != 0)
                {val += tmp_sum;}

                tmp_sum =__shfl_up_sync(0xffffffff,val,16);
                if(lane_id /16 != 0)
                {val += tmp_sum;}

                int judge = d_sortrowindex[j];
                    
                judge = judge-1;
                double scan_sum = __shfl_down(val,judge);
                    
                val_tmp = scan_sum - val +val_tmp;
                    
                if(d_sortrowindex[j] !=0)
                {
                    atomicAdd(&d_y[d_rowidx], val_tmp);
                }
            }
            break;
            
        }
    }

}


void FastLoad_spmv(char *filename,
                slide_matrix *matrix,
                int nnz,
                int colA,
                int rowA,
                double *x,
                double *y,
                double *y_golden)
{

    int tilenum = matrix->tilenum;
    int *tile_ptr = matrix->tile_ptr;
    //int *tile_tall = matrix->tile_tall;
    int *tile_len = matrix->tile_len;
    int *tile_colidx = matrix->tile_colidx;
    int *tile_format = matrix->tile_format;
    int *sortrowidx = matrix->sortrowidx;
    double *sortval = matrix->sortval;

//    int *countrow = matrix->countrow;
//    int *segmentoffset = matrix->segmentoffset;
    int segsum = matrix->segsum;
    int *sortrowindex = matrix->sortrowindex;


    int *d_tile_ptr;
    //int *d_tile_tall;
    int *d_tile_len;
    int *d_tile_colidx;
    int *d_format;
    int *d_sortrowidx;
    double *d_sortval;

//    int *d_countrow;
//    int *d_segmentoffset;
    int *d_sortrowindex;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_ptr, (tilenum + 1) * sizeof(int)));
    //CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_tall, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_len, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_colidx, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_format, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortrowidx,(nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortval,(nnz)*sizeof(double)));

//    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_countrow,(tilenum+1)*sizeof(int)));
//    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_segmentoffset,(segsum+1)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortrowindex,(nnz)*sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_tile_ptr, tile_ptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice));
    //CHECK_CUDA_ERROR(cudaMemcpy(d_tile_tall, tile_tall, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tile_len, tile_len, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tile_colidx, tile_colidx, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_format, tile_format, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortrowidx, sortrowidx, (nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortval, sortval, (nnz) * sizeof(double), cudaMemcpyHostToDevice));

//    CHECK_CUDA_ERROR(cudaMemcpy(d_countrow, countrow, (tilenum+1) * sizeof(int), cudaMemcpyHostToDevice));
//    CHECK_CUDA_ERROR(cudaMemcpy(d_segmentoffset, segmentoffset, (segsum+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortrowindex, sortrowindex, (nnz) * sizeof(int), cudaMemcpyHostToDevice));

    double *d_x;
    double *d_y;

    cudaMalloc((void **)&d_x, colA * sizeof(double));
    cudaMalloc((void **)&d_y, rowA * sizeof(double));

    cudaMemcpy(d_x, x, colA * sizeof(double), cudaMemcpyHostToDevice);
    

    int num_threads = slidesize *warpperblock;
    int num_blocks = ceil(( double)tilenum / (double)warpperblock); 


    FastLoad___kernel<<<num_blocks, num_threads>>>(rowA,colA,nnz,
                                                   tilenum,
                                                   d_tile_ptr,
                                                   //d_tile_tall,
                                                   d_tile_len,
                                                   d_tile_colidx,
                                                   d_format,
                                                   //d_countrow,
                                                   //d_segmentoffset,
                                                   d_sortrowindex,
                                                   d_sortrowidx,
                                                   d_sortval,
                                                   d_x,
                                                   d_y);


    timeval t1, t2;
    double time_cuda_spmv_base = 0;
    for(int i =0 ; i<100;i++)
    {
        int num_threads = slidesize *warpperblock;
        int num_blocks = ceil(( double)tilenum / (double)warpperblock);
        cudaMemset(d_y, 0, rowA * sizeof(double));
        gettimeofday(&t1, NULL);

        FastLoad___kernel<<<num_blocks, num_threads>>>(rowA,colA,nnz,
                                                   tilenum,
                                                   d_tile_ptr,
                                                   //d_tile_tall,
                                                   d_tile_len,
                                                   d_tile_colidx,
                                                   d_format,
                                                   //d_countrow,
                                                   //d_segmentoffset,
                                                   d_sortrowindex,
                                                   d_sortrowidx,
                                                   d_sortval,
                                                   d_x,
                                                   d_y);
        
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        time_cuda_spmv_base += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }
    time_cuda_spmv_base /= 100;
    double gflops = 2 * (double)nnz * 1.0e-6 / time_cuda_spmv_base;


    
    CHECK_CUDA_ERROR(cudaMemcpy(y, d_y, rowA * sizeof(double), cudaMemcpyDeviceToHost));

    int error_count_slide_cuda = 0;
    for (int i = 0; i < rowA; i++)
        if (abs(y_golden[i] - y[i]) > 0.01 * abs(y[i]))
        {
            error_count_slide_cuda++;
            
        }

    if (error_count_slide_cuda == 0)
    {
        printf("Check FASTLOAD GPU PASS! (%f ms , %4.2f GFlops)\n", time_cuda_spmv_base, gflops);
        FILE *fout = fopen("TimeResult/results_slidecuda.txt", "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "%s m %d n %d nnz %d FastLoad %f gflops %f\n",
                filename,rowA,colA,nnz,time_cuda_spmv_base, gflops);
        fclose(fout);
    }
    else
    {
        printf("Check NO PASS! error_count_slide_cuda = %d \n", error_count_slide_cuda);
        FILE *fout = fopen("TimeResult/results_slidecuda.txt", "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "error mtx: %s\n",
                                  filename);
        fclose(fout);
    }

    CHECK_LAST_CUDA_ERROR();

    cudaFree(d_tile_ptr);
    //cudaFree(d_tile_tall);
    cudaFree(d_tile_len);
    cudaFree(d_tile_colidx);
    cudaFree(d_format);
    cudaFree(d_sortrowidx);
    cudaFree(d_sortval);

//    cudaFree(d_countrow);
//    cudaFree(d_segmentoffset);
    cudaFree(d_sortrowindex);




}

