import os
import csv

output_dir = 'output'
csv_file = 'profiling_data.csv'

# Define the headers for the CSV file
headers = [
    'sparsity', 'matrix_size', 'M', 'N', 'nnz', 'CSC_SpMV_Time', 'Memory_H2D_Time',
    'Kernel_Time', 'Memory_D2H_Time', 'GFLOPS', 'FastLoad_sort_time',
    'FastLoad_format_transform_time', 'FastLoad_format_classification_time',
    'FastLoad_total_preprocessing_time', 'FastLoad_GPU_PASS_time', 'FastLoad_GFLOPS',
    'FastLoad_memcpy_H2D_time', 'FastLoad_memcpy_D2H_time'
]

data = []

# Open the CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

    # Iterate over the files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()

                # Extract the sparsity and matrix size from the filename
                parts = filename.split('_')
                sparsity = parts[1]
                matrix_size = parts[3].split('.')[0]

                # Extract the relevant information from the file
                M = lines[0].split(': ')[1].strip().split(',')[0]
                N = lines[0].split(': ')[2].strip()
                nnz = lines[1].split(': ')[1].strip()
                CSC_SpMV_Time = lines[3].split(': ')[1].strip().split(' ')[0]
                Memory_H2D_Time = lines[4].split(': ')[1].strip().split(' ')[0]
                Kernel_Time = lines[5].split(': ')[1].strip().split(' ')[0]
                Memory_D2H_Time = lines[6].split(': ')[1].strip().split(' ')[0]
                GFLOPS = lines[7].split(': ')[1].strip()
                FastLoad_sort_time = lines[10].split(': ')[1].strip().split(' ')[0]
                FastLoad_format_transform_time = lines[11].split(': ')[1].strip().split(' ')[0]
                FastLoad_format_classification_time = lines[12].split(': ')[1].strip().split(' ')[0]
                FastLoad_total_preprocessing_time = lines[13].split(': ')[1].strip().split(' ')[0]
                FastLoad_GPU_PASS_time = lines[14].split('(')[1].split(' ')[0]
                FastLoad_GFLOPS = lines[14].split(',')[1].strip().split(' ')[0]
                FastLoad_memcpy_H2D_time = lines[15].split(': ')[1].strip().split(' ')[0]
                FastLoad_memcpy_D2H_time = lines[16].split(': ')[1].strip().split(' ')[0]

                # Append the extracted information to the data list
                data.append([
                    sparsity, matrix_size, M, N, nnz, CSC_SpMV_Time, Memory_H2D_Time,
                    Kernel_Time, Memory_D2H_Time, GFLOPS, FastLoad_sort_time,
                    FastLoad_format_transform_time, FastLoad_format_classification_time,
                    FastLoad_total_preprocessing_time, FastLoad_GPU_PASS_time, FastLoad_GFLOPS,
                    FastLoad_memcpy_H2D_time, FastLoad_memcpy_D2H_time
                ])

    # Sort the data by sparsity and then by matrix size
    data.sort(key=lambda x: (float(x[0]), int(x[1])))

    # Round all floating point values to 3 decimal places and write to CSV
    for row in data:
        rounded_row = [
            row[0], row[1], row[2], row[3], row[4],
            f"{float(row[5]):.3f}", f"{float(row[6]):.3f}", f"{float(row[7]):.3f}",
            f"{float(row[8]):.3f}", f"{float(row[9]):.3f}", f"{float(row[10]):.3f}",
            f"{float(row[11]):.3f}", f"{float(row[12]):.3f}", f"{float(row[13]):.3f}",
            f"{float(row[14]):.3f}", f"{float(row[15]):.3f}", f"{float(row[16]):.3f}",
            f"{float(row[17]):.3f}"
        ]
        writer.writerow(rounded_row)

