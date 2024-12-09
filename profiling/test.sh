#!/usr/bin/bash

sparsities=(10 20 30 40 50 60 70 80 90 95)
matrix_sizes=(1000 2000 4000 6000 8000 10000 12000)

# make output directory
mkdir -p output

for sparsity in ${sparsities[@]}; do
    for matrix_size in ${matrix_sizes[@]}; do
        echo "Running for sparsity: $sparsity, matrix size: $matrix_size"
        ./test $matrix_size $matrix_size $sparsity > output/sparsity_${sparsity}_size_${matrix_size}.txt
    done
done
