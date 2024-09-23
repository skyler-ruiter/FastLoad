
void FastLoad_cpu(slide_matrix *matrix,
               int nnz,
               int rowA,
               int colA,
               
               double *x,
               double *y,
               double *y_goldencsc)

{
    double *sortval = matrix->sortval;
    int *sortrowidx = matrix->sortrowidx;
    int *tile_ptr = matrix->tile_ptr;
    int *tile_len = matrix->tile_len;
    int *tile_colidx = matrix->tile_colidx;
    int tilenum = matrix->tilenum;

    for(int i=0;i<tilenum;i++)
    {
        int start = tile_ptr[i];
        int stop = tile_ptr[i+1];
        for(int j = start; j<stop;j++)
        {
            int len=tile_len[i];
            int colidx=tile_colidx[i] + (j-start)%len ;
            
            int rowidx = sortrowidx[j];
            y[rowidx] +=sortval[j] * x[colidx];
        }
    }

    int error_count_check_cpu = 0;
    for (int i = 0; i < rowA; i++)
    {
        if (y_goldencsc[i] !=y[i])
        {
            error_count_check_cpu++;
        }
    }
    if (error_count_check_cpu != 0)
    {
        printf("Check NO PASS! error_count_check_cpu= %d \n", error_count_check_cpu);
    }


}