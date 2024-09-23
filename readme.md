# FastLoad

**FastLoad** is an open-source code that optimizes sparse matrix-vector multiplication (SpMV) based on CSC format on GPUs. 


## Contact us

If you have any questions about running the code, please contact Jinyu Hu. 

E-mail: hujinyu@hnu.edu.cn

## Introduction

Sparse Matrix-Vector Multiplication (SpMV) on GPUs has gained significant attention because of SpMV's importance in modern applications and the increasing computing power of GPUs in the last decade. Previous studies have emphasized the importance of data loading for the overall performance of SpMV and demonstrated the efficacy of coalesced memory access in enhancing data loading efficiency. However, existing approaches fall far short of reaching the full potential of data loading on modern GPUs. In this paper, we propose an efficient algorithm called FastLoad, that speeds up the loading of both sparse matrices and input vectors of SpMV on modern GPUs. Leveraging coalesced memory access, FastLoad achieves high loading efficiency and load balance by sorting both the columns of the sparse matrix and elements of the input vector based on the number of non-zero elements while organizing non-zero elements in blocks to avoid thread divergence. FastLoad takes the Compressed Sparse Column (CSC) format as an implementation case to prove the concept and gain insights.


## Execution of FastLoad
1 NVIDIA GPU with compute capability at least 3.5 (NVIDIA 3090ti as tested) * NVIDIA nvcc CUDA compiler and cuSPARSE library, both of which are included with CUDA Toolkit (CUDA v11.1 as tested)
2 Ubuntu 22.04,
Our test programs currently support input files encoded using the matrix market format.

1. Set CUDA path in the Makefile

2. The command 'make' generates an executable file 'test' for double/single precision.
> **cd src**

> **mkdir TimeResult**

> **mkdir PreProcess**

> **make**

3. Run (give a example matrix)  
> **./test ../Data/1138_bus.mtx**

The result will shown in fold TimeResult and PreProcess.
