#include "coo2csc.h"
#include "sys/time.h"
#include "FastLoad_CPU.h"
#include "FastLoad_GPU.h"
#include "CSCSpMV.h"
#include "csr2csc_cuda.h"
#include "formatTransform_GPU.h"
#include "ColSort_GPU.h"

#include "cusparse_cuda.h"
#include <iostream>


int main(int argc, char ** argv)
{
      
    if(argc <2)
    {
        printf("error order\n");
        return 0;
    }

    int rowA,colA,nnz;
    int isSymmetricA;
    double *csrval;
    int *csrrowptr;
    int *csrcolidx;

    char *filename;
    filename = argv[1];
    printf("------------%s-------------\n", filename);


//------------------------------------read matrix----------------------------------------------------------

    struct timeval read1,read2;
    double readtime;
    gettimeofday(&read1,NULL);
    
    mmio_allinone(&rowA, &colA, &nnz, &isSymmetricA, &csrrowptr, &csrcolidx, &csrval ,filename);

    gettimeofday(&read2,NULL);
    readtime = (read2.tv_sec - read1.tv_sec) * 1000.0 + (read2.tv_usec - read1.tv_usec) / 1000.0;
    
//---------------------------------------------------------------------------------------------------


    std::string filePath(filename);
    size_t pos = filePath.find_last_of("/\\");
    std::string fileName;
    if (pos != std::string::npos) 
    {
        fileName = filePath.substr(pos + 1);
    } 
    else 
    {
        fileName = filePath;
    }
    size_t dotPos = fileName.find_last_of('.');
    if (dotPos != std::string::npos) 
    {
        fileName = fileName.substr(0, dotPos);
    }

    filename = new char[fileName.length() + 1];
    std::strcpy(filename, fileName.c_str());

    for (int i = 0; i < nnz; i++)
	{csrval[i] = i % 10;}
   
    printf("read success,input matrix A :(%i,%i) nnz =%i  \n",rowA,colA,nnz);





//-------------------------------------------------------csr2csc---------------------------------------------------------------------------

    int *cscrowidx = (int *)malloc(sizeof(int) * (nnz));
    memset(cscrowidx, 0, sizeof(int)*(nnz));
    int *csccolptr = (int *)malloc(sizeof(int) * (colA +1));
    memset(csccolptr, 0, sizeof(int)*(colA+1));
    double *cscval = (double *)malloc(sizeof(double)* (nnz));
    memset(cscval, 0, sizeof(double)*(nnz));
    int *nnzpercol=(int *)malloc(sizeof(int)*(colA));
    memset(nnzpercol, 0 ,sizeof(int)*(colA));

/* 
    struct timeval transpose1,transpose2;
    double transposetime;
    double csr2cscTime=0;
    gettimeofday(&transpose1,NULL);

    csr2csc(nnz,rowA,colA,csrcolidx,csrrowptr,csrval,&cscrowidx,&csccolptr,&nnzpercol,&cscval); //CPU

    gettimeofday(&transpose2,NULL);
    transposetime = (transpose2.tv_sec - transpose1.tv_sec) * 1000.0 + (transpose2.tv_usec - transpose1.tv_usec) / 1000.0;
*/

    double csr2cscTime=0;
    csr_to_csc(csr2cscTime, rowA, colA, nnz, csrval, csrrowptr, csrcolidx, cscval, cscrowidx, csccolptr, nnzpercol);





    double *x= (double *)malloc(sizeof(double)*colA);
    memset(x,0,sizeof(double)*colA);
    for(int i=0;i<colA;i++)
    {
        x[i]=i%10;
    }

    double *y_goldencsr = (double *)malloc(sizeof(double) * rowA);
    memset(y_goldencsr,0,sizeof(double)*rowA);
	for (int i = 0; i < rowA; i++)
	{
		double sum = 0;
		for (int j = csrrowptr[i]; j < csrrowptr[i+1]; j++)
		{
			sum += csrval[j] * x[csrcolidx[j]];
		}
		y_goldencsr[i] = sum;
	}

    double *y_goldencsc = (double *)malloc(sizeof(double) * rowA);
    memset(y_goldencsc,0,sizeof(double)*rowA);
	for (int i = 0; i < colA; i++)
	{
		for (int j = csccolptr[i]; j < csccolptr[i+1]; j++)
		{
			int row_tmp = cscrowidx[j];
            double val_tmp = cscval[j];
            y_goldencsc[row_tmp] += val_tmp * x[i];
		}
	}

    int error_count_check = 0;
    for (int i = 0; i < rowA; i++)
        if (y_goldencsc[i] !=y_goldencsr[i])
        {
            error_count_check++;
        }

    if (error_count_check != 0)
        {
            printf("Check NO PASS! error_count_check_csc_csr= %d \n", error_count_check);            
        }



    slide_matrix *matrixA = (slide_matrix *)malloc(sizeof(slide_matrix));

    int *sortrowidx_tmp = (int *)malloc(sizeof(int)*nnz);
    memset(sortrowidx_tmp,0,sizeof(int)*nnz);
    double *sortval_tmp = (double *)malloc(sizeof(double)*nnz);
    memset(sortval_tmp,0,sizeof(double)*nnz);
    int *sortnnz_tmp= (int *)malloc(sizeof(int)*(colA));
    memset(sortnnz_tmp,0,sizeof(int)*colA);
    double *sortx = (double *)malloc(sizeof(double)*colA);  
    memset(sortx,0,sizeof(double)*colA);


//----------------------------------------------------collumn sort----------------------------------------------------------------------------------------
    double timeForSort =0;

    ColSort(timeForSort,
            rowA,
            colA,
            nnz,
            nnzpercol,
            csccolptr,
            cscrowidx,
            cscval,             
            sortrowidx_tmp,
            sortval_tmp,
            sortnnz_tmp,              
            x,
            sortx);
            

/*
    struct timeval collumsort1,collumsort2;
    double colsorttime;
    gettimeofday(&collumsort1,NULL);

    col_sort(colA,
             nnzpercol,
             csccolptr,
             cscrowidx,
             cscval,
              
             sortrowidx_tmp,
             sortval_tmp,
             sortnnz_tmp,
             
             x,
             sortx,
             
             nnz); //CPU

    gettimeofday(&collumsort2,NULL);
    colsorttime = (collumsort2.tv_sec - collumsort1.tv_sec) * 1000.0 + (collumsort2.tv_usec - collumsort1.tv_usec) / 1000.0;
*/




//-------------------------------------------------matrix reorder------------------------------------------------------------------------------------------------


    int h_count;   
    double timeFormatTran=0;
    double timeFortmatClas=0;

   
    formatTransform(timeFormatTran,
                    timeFortmatClas,
                    matrixA,
                    sortrowidx_tmp,
                    sortval_tmp,
                    sortnnz_tmp,
                    nnz,
                    rowA,
                    colA,
                    h_count); //GPU


/*
    struct timeval formattransation1,formattransation2;
    double formattransationtime;
    gettimeofday(&formattransation1,NULL);

    formattransation(matrixA,
                     sortrowidx_tmp,
                     sortval_tmp,
                     sortnnz_tmp,
                     
                     nnz,
                     rowA,
                     colA,
                     h_count); //CPU

    gettimeofday(&formattransation2,NULL);
    formattransationtime = (formattransation2.tv_sec - formattransation1.tv_sec) * 1000.0 + (formattransation2.tv_usec - formattransation1.tv_usec) / 1000.0;
*/


    free(nnzpercol);
    free(sortrowidx_tmp);
    free(sortval_tmp);
    free(sortnnz_tmp);

/*
    struct timeval classification1,classification2;
    double classificationtime;
    gettimeofday(&classification1,NULL);  

    tile_classification(matrixA,
                        nnz,rowA,colA); //CPU only
                          
    gettimeofday(&classification2,NULL);
    classificationtime = (classification2.tv_sec - classification1.tv_sec) * 1000.0 + (classification2.tv_usec - classification1.tv_usec) / 1000.0;
*/




    double d_overalltime = csr2cscTime + timeForSort + timeFormatTran + timeFortmatClas;
    double d_percent1 = csr2cscTime / d_overalltime;
    double d_percent2 = timeForSort / d_overalltime;
    double d_percent3 = timeFormatTran / d_overalltime;
    double d_percent4 = timeFortmatClas / d_overalltime;    


    double *y = (double *)malloc(sizeof(double)*rowA);
    memset(y,0,sizeof(double)*rowA);
   
    struct timeval time1,time2;
    double cputime;
    gettimeofday(&time1,NULL);

    FastLoad_cpu(matrixA,
              nnz,
              rowA,
              colA,

              sortx,
              y,
              y_goldencsc);

    gettimeofday(&time2,NULL);
    cputime = (time2.tv_sec - time1.tv_sec)*1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;


    FILE *fout3 = fopen("PreProcess/preprocesstime_GPU.txt", "a");
    if (fout3 == NULL)
    printf("Writing results fails.\n");
    fprintf(fout3, "%s nnz %d mtxRead %f overall %f csr2csc %f %f colSort %f %f formatTran %f %f blkclassi %f %f cpu_time %f \n",
            filename , nnz , readtime , d_overalltime,csr2cscTime , d_percent1 , timeForSort , d_percent2 , timeFormatTran , d_percent3, timeFortmatClas, d_percent4,cputime);
    fclose(fout3);


    memset(y,0,sizeof(double)*rowA);
    csc_spmv(filename,
             rowA,
             colA,
             nnz,
             csccolptr,
             cscrowidx,
             cscval,
              
             x,
             y,
             y_goldencsr);

    memset(y,0,sizeof(double)*rowA);
    call_cusparse_spmv(filename,
                       nnz,
                       rowA,
                       colA,
                       csrrowptr,
                       csrcolidx,
                       csrval,
                       x,
                       y,
                       y_goldencsr);

    memset(y,0,sizeof(double)*rowA);
    FastLoad_spmv(filename,
               matrixA,
               nnz,
               colA,
               rowA,
               sortx,
               y,
               y_goldencsr);

 

}


