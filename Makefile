#compilers
CC=nvcc

#GLOBAL_PARAMETERS
MAT_VAL_TYPE = double

PAPI_INC = /home/skyler/papi-install/include
PAPI_LIB = /home/skyler/papi-install/lib

NVCC_FLAGS = -O3 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda-11.1


#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I/home/usr/NVIDIA_CUDA-11.1_Samples/common/inc 

CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64  -lcudart -lcusparse
LIBS = $(CUDA_LIBS) 

#options
OPTIONS = -Xcompiler -fopenmp -O3 #-std=c99

make:
	$(CC) $(NVCC_FLAGS) test.cu -o test  $(INCLUDES) $(LIBS) $(OPTIONS) -D MAT_VAL_TYPE=$(MAT_VAL_TYPE)
