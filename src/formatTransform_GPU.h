#include <cuda_runtime_api.h>
#include <iostream>
#include <cub/cub.cuh>
using namespace cub;
using namespace std;


__global__ void SegmentReduce(int *nnzpercolinput, int *nnzpercoloutput, int nnzpercolnumber)
{
    nnzpercoloutput[0] = nnzpercolinput[0];
    int tid =blockDim.x * blockIdx.x +threadIdx.x;
    if(tid < nnzpercolnumber && tid >0)
    {
        nnzpercoloutput[tid] = nnzpercolinput[tid] - nnzpercolinput[tid-1];
    }

}

//---------------------------------------------------------for transform----------------------------------------------------------------------------------------------------

__global__ void prepare(int *colReduce, int *colTmp, int *tileCol, int *tileNnz, int *tileTall, int *Auxiliaryarray, int size, int *count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_j=0;
      
    if(colReduce[0] != 0)
    {
        tileCol[0] = 0;
        tileNnz[0] = 0;
        tileTall[0] = colReduce[0];

    }



    if(idx>0 && idx <size)
    {
        int local_count=0;
        if(colReduce[idx] !=0)
        {
            for(int i = 0; i<idx;i++)
            {
                if(colReduce[i] !=0)
                {Auxiliaryarray[idx] = Auxiliaryarray[idx] +1;}
                else
                {Auxiliaryarray[idx] = Auxiliaryarray[idx] +0;}
            }
            d_j = Auxiliaryarray[idx];
            tileCol[d_j] = idx;
            tileNnz[d_j] = colTmp[idx-1];
            tileTall[d_j] = colReduce[idx];
            local_count = 1;

        }  
    }

}


__global__ void countNonZeroElements(const int *d_array, int size, int *d_count) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localCount = 0;

    if (tid < size) {
        if (d_array[tid] != 0) {
            localCount = 1;
        }
    }
    atomicAdd(d_count, localCount);
}

__global__ void countNonZeroElements2(int *d_array, int *auxiArray, int size) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) 
    {
        if (d_array[tid] != 0) 
        {
            auxiArray[tid] = 1;
        }
    }
}

__global__ void countTileNum(int *tileCol, int *tileTall, int d_count, int *d_tileCount, int colA)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //int h_couttt = *d_count;
    //if(tid < h_couttt)
    if(tid < d_count)
    {
        int j = tileCol[tid];
        int t = tileTall[tid];
        int tile = (colA - j) %32 ==0? (colA - j) / 32 : (colA - j) / 32 +1;
        int tilesum = tile *t ;
        atomicAdd(d_tileCount, tilesum);
    }
}

__global__ void countTileNum2(int *tileCol, int *tileTall, int d_count, int *auxiArray, int colA)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < d_count)
    {
        int j = tileCol[tid];
        int t = tileTall[tid];
        int tile = (colA - j) %32 ==0? (colA - j) / 32 : (colA - j) / 32 +1;
        int tilesum = tile *t;
        auxiArray[tid] = tilesum;        
    }
}

__global__ void auxiliaryTrans(int *tileCol, int *tileTall, int *tileNnz,int *tilePtr, int *nnzPerTileSort,int *tileLen, int *tileColidx, int *tileTallPre, int colA, int countCol )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;     
    if(tid < countCol)
    {
        int ii = tileCol[tid];
        int tile = (colA - ii) %32 ==0? (colA - ii) / 32 : (colA - ii) / 32 +1;
        int ti = tileTall[tid];
        int tmp = 0;

        for(int i = 0; i < tid; i++)
        {
            int ji = tileCol[i];
            int t = tileTall[i];
            int tilei = (colA - ji) %32 ==0? (colA - ji) / 32 : (colA - ji) / 32 +1;
            tmp =  tmp + tilei *t ;
        }

        for(int j =0; j<tile; j++)
        {
            int len = j == tile-1? colA - ii - (tile-1)*32 : 32;
            int colidxTmp = ii +32 * j;
            for(int jx = 0; jx<ti; jx ++)
            { 
                tilePtr[tmp] = len;
                nnzPerTileSort[tmp] = len;
                tileLen[tmp] = len;
                tileColidx[tmp] = colidxTmp;
                tileTallPre[tmp]= tileNnz[tid] + jx;
                tmp ++;
                //printf("tid : %d ii:%d, tmp :%d coltmp:%d \n",tid, ii, tmp, colidxTmp);
            }

        }

    }
    
}

__global__ void dataTrans(int tileNum,int *tileColidx, int *tileLen,int *tileTallPre,int *tilePtr,int *nnzColPtr,int *sortRowidxTmp, double *sortValTmp, int *sortRowIdx, double *sortVal )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < tileNum)
    {
        int colidxStart = tileColidx[tid];
        int lenTmp = tileLen[tid];
        int colidxStop = colidxStart + lenTmp;
        int tallStart = tileTallPre[tid];
        for(int ri = colidxStart; ri <colidxStop; ri++)
        {
            int j = tilePtr[tid];
            int rii = nnzColPtr[ri] + tallStart;
            sortVal[j] = sortValTmp[rii];
            sortRowIdx[j] = sortRowidxTmp[rii];
            tilePtr[tid]++;
            //printf("idx:%d,j:%d, value:%f \n",tid,j,sortVal[j]);
        }
    }    
}

__global__ void PtrReduce(int tileNum,int *nnzPerTileSort, int *tilePtr)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    tilePtr[0] = 0;
    if(tid >0 && tid<tileNum)
    {
        tilePtr[tid] = tilePtr[tid] - nnzPerTileSort[tid];

    }



}


__global__ void Pre4CountRow(int tileNum,int *tilePtr, int *rowIdxTmp, int *countrow)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < tileNum)
    {
        countrow[tid] = 1;
        int start = tilePtr[tid];
        int stop = tilePtr[tid+1];
        int compare = rowIdxTmp[start];
        for(int j = start+1; j<stop; j++)
        {
            if(rowIdxTmp[j] != compare)
            {
                compare = rowIdxTmp[j];
                countrow[tid]++;

            }
        }
    }

}

__global__ void Pre4segOff(int tileNum,int *tilePtr, int *rowIdxTmp, int *countrow,int *segOffset, int segsum)
{
    segsum = countrow[tileNum];
    //printf("%d\n",segsum);
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < tileNum)
    {
        int start = tilePtr[tid];
        int stop = tilePtr[tid+1];
        int segpoint = countrow[tid];
        int compare = rowIdxTmp[start];
        for(int j = start+1; j<stop; j++)
        {
            if(rowIdxTmp[j] == compare)
            {
                segOffset[segpoint]++;
            }
            else
            {
                compare = rowIdxTmp[j];
                segpoint++;
            }
        }
    } 
}
__global__ void Pre4addone(int tileNum,int *countrow,int *segOffset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int segsum = countrow[tileNum];
    if(tid < segsum)
    {
        segOffset[tid]++;
    }
    
}

__global__ void Pre4SegIndex(int tileNum,int *countrow,int *segOffset, int *segOffsetTmp, int *rowsegIndex)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<tileNum)
    {
        int start = countrow[tid];
        int stop = countrow[tid +1];
        for(int j = start; j <stop; j++)
        {
            int index = segOffsetTmp[j];
            int targetval = segOffset[j];
            rowsegIndex[index] = targetval;
                //printf("index: %d, rowseg: %d \n", tid, rowsegIndex[tid]);
        }
    }
}

__global__ void Pre4tileclass(int tileNum,int *countrow, int *tileLen, int *tileFormat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    {
        int lent = tileLen[tid];
        int start = countrow[tid];
        int stop = countrow[tid+1];
        int judegd = stop -start;
        for(int j = start; j<stop;j++)
        {
            if(judegd != 1 && lent ==32 )
            {
                tileFormat[tid] =2;
            }
            else if (judegd ==1 && lent ==32)
            {
                tileFormat[tid] =1;
            }
        }
        
    }   

}



void formatTransform(double &timeFormatTran,
                     double &timeFormatclas,
                     slide_matrix *matrix,
                     int *sortrowidx_tmp,
                     double *sortval_tmp,
                     int *sortnnz_tmp,
                     int nnz,
                     int rowA,
                     int colA,
                     int &h_count)
{
    int *d_nnzcolReduce;
    int *d_nnzcolPtr;

    int *d_fortilecol;
    int *d_fortilennz;
    int *d_fortiletall;


    int *d_sortnnzTmp;

    int *d_auxiliaryForcount;
    int *d_countcol;
    int *d_tilenumcount;


    int tilenum;

    int *d_sortrowidxTmp;
    double *d_sortvalTmp;

    int *d_sortrowidx;
    double *d_sortval;
    int *d_auxiArray;
    int *d_auxiArray2;
    int h_count2;

    int *d_auxiArray3;
    int *d_auxiArray4;
    int tilenum2;
//-------------------------

    int *d_tile_ptr;
    int *d_tile_len;
    int *d_tile_colidx;

    int *d_tileTallPre;
    int *d_nnzPerTileSort;
    int *d_tile_ptr_final;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_ptr, (nnz) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_len, (nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_colidx, (nnz)*sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tileTallPre, (nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_nnzPerTileSort, (nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_ptr_final, (nnz) * sizeof(int)));

//-------------------
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_nnzcolReduce, (colA) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_nnzcolPtr, (colA + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_fortilecol, (colA) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_fortilennz, (colA) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_fortiletall, (colA) * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortnnzTmp, (colA+1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortrowidxTmp, (nnz) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortvalTmp, (nnz) * sizeof(double)));


    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortrowidx, (nnz) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortval, (nnz) * sizeof(double)));



    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_auxiliaryForcount, (colA) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_auxiliaryForcount, 0, colA * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_countcol, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_countcol, 0, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tilenumcount, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_tilenumcount, 0, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_auxiArray, (colA+1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_auxiArray, 0, colA+1 * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_auxiArray2, (colA+1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_auxiArray2, 0, colA+1 * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_auxiArray3, (colA+1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_auxiArray3, 0, colA+1 * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_auxiArray4, (colA+1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_auxiArray4, 0, colA+1 * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_sortnnzTmp, sortnnz_tmp, (colA+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortrowidxTmp, sortrowidx_tmp, (nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortvalTmp, sortval_tmp, (nnz) * sizeof(double), cudaMemcpyHostToDevice));

    timeval tFormatT1, tFormatT2;
    timeFormatTran = 0;

    gettimeofday(&tFormatT1, NULL);
    const int blockSize=256;
    const int numBlocksColReduce = (colA+ blockSize -1 ) / blockSize;
    SegmentReduce<<<numBlocksColReduce, blockSize>>>(d_sortnnzTmp, d_nnzcolReduce, colA);



    void *d_temp_storage1 = nullptr;
    size_t temp_storage_bytes1 = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, d_sortnnzTmp, d_nnzcolPtr, colA+1,0);

    cudaMalloc(&d_temp_storage1, temp_storage_bytes1); 

    cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, d_sortnnzTmp, d_nnzcolPtr, colA+1,0);




    const int numBlocksPrepare = (colA + blockSize -1) / blockSize;
    prepare<<<numBlocksPrepare, blockSize>>>(d_nnzcolReduce, d_sortnnzTmp, d_fortilecol, d_fortilennz, d_fortiletall,d_auxiliaryForcount, colA,d_countcol);




    const int numBlockscountcol = (colA + blockSize -1) / blockSize;

    countNonZeroElements<<<numBlockscountcol, blockSize>>>(d_nnzcolReduce, colA, d_countcol);

    CHECK_CUDA_ERROR(cudaMemcpy(&h_count, d_countcol, sizeof(int), cudaMemcpyDeviceToHost));
     
    const int numBlockscountTile = (h_count + blockSize -1) / blockSize;
    countTileNum<<<numBlockscountTile, blockSize>>>(d_fortilecol, d_fortiletall, h_count, d_tilenumcount, colA);    

    CHECK_CUDA_ERROR(cudaMemcpy(&tilenum, d_tilenumcount, sizeof(int), cudaMemcpyDeviceToHost));


    
    const int numBlocksTilePre = (h_count + blockSize -1) / blockSize;
    auxiliaryTrans<<<numBlocksTilePre, blockSize>>>(d_fortilecol, d_fortiletall, d_fortilennz,d_tile_ptr, d_nnzPerTileSort,d_tile_len, d_tile_colidx, d_tileTallPre, colA, h_count );



    void *d_temp_storage2 = nullptr;
    size_t temp_storage_bytes2 = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage2, temp_storage_bytes2, d_tile_ptr, d_tile_ptr_final, tilenum+1,0);
    cudaMalloc(&d_temp_storage2, temp_storage_bytes2); 
    cub::DeviceScan::ExclusiveSum(d_temp_storage2, temp_storage_bytes2, d_tile_ptr, d_tile_ptr_final, tilenum+1,0);


    const int numBlockstrans = (tilenum + blockSize-1) / blockSize;
    dataTrans<<<numBlockstrans,blockSize>>>(tilenum, d_tile_colidx, d_tile_len, d_tileTallPre, d_tile_ptr_final, d_nnzcolPtr, d_sortrowidxTmp, d_sortvalTmp, d_sortrowidx, d_sortval );
    PtrReduce<<<numBlockstrans,blockSize>>>(tilenum,d_nnzPerTileSort, d_tile_ptr_final);


    gettimeofday(&tFormatT2, NULL);
    timeFormatTran= (tFormatT2.tv_sec - tFormatT1.tv_sec) * 1000.0 + (tFormatT2.tv_usec - tFormatT1.tv_usec) / 1000.0;    



//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    int *d_countrow;
    int *d_countrowFi;
    int *d_segOff;
    int *d_segOffTmp;
    int d_segsum;

    int *d_rowsegIndex;
    int *d_tileFormat;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_countrow, (tilenum+1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_countrowFi, (tilenum+1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_segOff, (nnz) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_segOffTmp, (nnz+1) * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_rowsegIndex, (nnz) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_rowsegIndex, 0, nnz * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tileFormat, (tilenum) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_tileFormat, 0, tilenum * sizeof(int)));


    timeFormatclas=0;
    timeval tFormatC1, tFormatC2;
    gettimeofday(&tFormatC1, NULL);


    Pre4CountRow<<<numBlockstrans,blockSize>>>(tilenum,d_tile_ptr_final, d_sortrowidx, d_countrow);

    void *d_temp_storage3 = nullptr;
    size_t temp_storage_bytes3 = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage3, temp_storage_bytes3, d_countrow, d_countrowFi, tilenum+1,0);
    cudaMalloc(&d_temp_storage3, temp_storage_bytes3); 
    cub::DeviceScan::ExclusiveSum(d_temp_storage3, temp_storage_bytes3, d_countrow, d_countrowFi, tilenum+1,0);

    Pre4segOff<<<numBlockstrans,blockSize>>>(tilenum,d_tile_ptr_final, d_sortrowidx, d_countrowFi,d_segOff, d_segsum);

    const int numBlocks4seg = (nnz + blockSize-1) / blockSize;
    Pre4addone<<<numBlocks4seg,blockSize>>>(tilenum,d_countrowFi,d_segOff);

    void *d_temp_storage4 = nullptr;
    size_t temp_storage_bytes4 = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage4, temp_storage_bytes4, d_segOff, d_segOffTmp, nnz+1,0);
    cudaMalloc(&d_temp_storage4, temp_storage_bytes4); 
    cub::DeviceScan::ExclusiveSum(d_temp_storage4, temp_storage_bytes4, d_segOff, d_segOffTmp, nnz+1,0);

    Pre4SegIndex<<<numBlockstrans,blockSize>>>(tilenum, d_countrowFi, d_segOff, d_segOffTmp, d_rowsegIndex);

    Pre4tileclass<<<numBlockstrans,blockSize>>>(tilenum,d_countrowFi, d_tile_len, d_tileFormat);

    gettimeofday(&tFormatC2, NULL);
    timeFormatclas= (tFormatC2.tv_sec - tFormatC1.tv_sec) * 1000.0 + (tFormatC2.tv_usec - tFormatC1.tv_usec) / 1000.0;    



    matrix->tile_ptr = (int *)malloc(( 1 + tilenum)*sizeof(int));
    memset(matrix->tile_ptr,0,(1+tilenum)*sizeof(int));
    int *h_tilePtr = matrix->tile_ptr;   

    matrix->tile_len = (int *)malloc((tilenum)*sizeof(int));
    memset(matrix->tile_len,0,(tilenum)*sizeof(int));
    int *h_tileLen = matrix->tile_len;

    matrix->tile_colidx = (int *)malloc((tilenum)*sizeof(int));
    memset(matrix->tile_colidx,0,(tilenum)*sizeof(int));
    int *h_tileColidx = matrix->tile_colidx;



    matrix->sortval = (double *)malloc(nnz * sizeof(double));
    memset(matrix->sortval,0,nnz*sizeof(double));
    double *h_sortvaltest = matrix ->sortval;


    matrix->sortrowidx = (int *)malloc(nnz * sizeof(int));
    memset(matrix->sortrowidx,0,nnz * sizeof(int));
    int *h_sortRowidx = matrix->sortrowidx;

//--------------------------------------------------------------------
    matrix->sortrowindex = (int *)malloc((nnz) * sizeof(int));
    memset(matrix->sortrowindex,0,(nnz) * sizeof(int));
    int *h_sortrowindex = matrix->sortrowindex;

    matrix->tile_format = (int *)malloc(tilenum * sizeof(int));
    memset(matrix->tile_format,0,tilenum*sizeof(int));
    int *h_tile_formatc = matrix->tile_format;

    CHECK_CUDA_ERROR(cudaMemcpy(h_tilePtr, d_tile_ptr_final,(tilenum+1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_tileLen, d_tile_len, tilenum * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_tileColidx, d_tile_colidx, tilenum * sizeof(int), cudaMemcpyDeviceToHost));    
    CHECK_CUDA_ERROR(cudaMemcpy(h_sortvaltest, d_sortval,(nnz) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_sortRowidx, d_sortrowidx,(nnz) * sizeof(int), cudaMemcpyDeviceToHost));
    matrix->tilenum = tilenum;
    CHECK_CUDA_ERROR(cudaMemcpy(h_sortrowindex, d_rowsegIndex,(nnz) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_tile_formatc, d_tileFormat,(tilenum) * sizeof(int), cudaMemcpyDeviceToHost));




    
    CHECK_CUDA_ERROR(cudaFree(d_nnzcolReduce));

    CHECK_CUDA_ERROR(cudaFree(d_nnzcolPtr));
    CHECK_CUDA_ERROR(cudaFree(d_fortilecol));
    CHECK_CUDA_ERROR(cudaFree(d_fortilennz));
    CHECK_CUDA_ERROR(cudaFree(d_fortiletall));


    CHECK_CUDA_ERROR(cudaFree(d_sortrowidxTmp));
    CHECK_CUDA_ERROR(cudaFree(d_sortvalTmp));
    CHECK_CUDA_ERROR(cudaFree(d_sortnnzTmp));

    CHECK_CUDA_ERROR(cudaFree(d_tile_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_tile_len));
    CHECK_CUDA_ERROR(cudaFree(d_tile_colidx));
    CHECK_CUDA_ERROR(cudaFree(d_sortval));
    CHECK_CUDA_ERROR(cudaFree(d_sortrowidx));

    CHECK_CUDA_ERROR(cudaFree(d_tileTallPre));
    CHECK_CUDA_ERROR(cudaFree(d_nnzPerTileSort));
    CHECK_CUDA_ERROR(cudaFree(d_tile_ptr_final));

    CHECK_CUDA_ERROR(cudaFree(d_auxiliaryForcount));
    CHECK_CUDA_ERROR(cudaFree(d_countcol));
    CHECK_CUDA_ERROR(cudaFree(d_tilenumcount));
    CHECK_CUDA_ERROR(cudaFree(d_auxiArray));
    CHECK_CUDA_ERROR(cudaFree(d_auxiArray2));
    CHECK_CUDA_ERROR(cudaFree(d_auxiArray3));
    CHECK_CUDA_ERROR(cudaFree(d_auxiArray4));   
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    CHECK_CUDA_ERROR(cudaFree(d_countrow));
    CHECK_CUDA_ERROR(cudaFree(d_countrowFi));
    CHECK_CUDA_ERROR(cudaFree(d_segOff));
    CHECK_CUDA_ERROR(cudaFree(d_segOffTmp));
    CHECK_CUDA_ERROR(cudaFree(d_rowsegIndex));
    CHECK_CUDA_ERROR(cudaFree(d_tileFormat));

    CHECK_CUDA_ERROR(cudaFree(d_temp_storage1));    
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage2));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage3));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage4));   
  
CHECK_LAST_CUDA_ERROR();
}


                