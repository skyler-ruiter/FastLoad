#!/bin/bash

# This script is used to run NCU tests on the FastLoad application.

sparsities=(10 30 50 70 90)
matrix_sizes=(1000 2000 4000 6000 8000 10000 12000)

# make output directory
mkdir -p ncu_output

for sparsity in ${sparsities[@]}; do
    for matrix_size in ${matrix_sizes[@]}; do
        echo "Running for sparsity: $sparsity, matrix size: $matrix_size"
        csc_output_file=ncu_output/csc_${sparsity}_${matrix_size}
        sudo env "PATH=$PATH" ncu -f --launch-count 5 -o $csc_output_file --set full -k regex:"CSC" ./test $matrix_size $matrix_size $sparsity 
        fastload_output_file=ncu_output/fastload_${sparsity}_${matrix_size}
        sudo env "PATH=$PATH" ncu -f --launch-count 5 -o $fastload_output_file --set full -k regex:"FastLoad" ./test $matrix_size $matrix_size $sparsity
    done
done

