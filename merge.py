## Merge separate .csv files into input & output

print('Imports')
import glob
import os
import pandas as pd

# Input to hdf5
print('Input files')
path = '/opt/uio/data/'
xn_filename = 'xn*.csv'
csv_file_list = glob.glob(os.path.join(path, xn_filename))
csv_file_list.sort()
csv_file_list

chunksize = 1000000
input_filename = os.path.join(path, 'input.hdf')

for csv_file_name in csv_file_list:
    print(csv_file_name)
    chunk_container = pd.read_csv(csv_file_name, chunksize=chunksize)
    for chunk in chunk_container:
        chunk.to_hdf(input_filename, key='df', mode="a", index=False)

# Output to hdf5
print('Output files')
yn_filename = 'yn*.csv'
csv_file_list = glob.glob(os.path.join(path, yn_filename))
csv_file_list.sort()
csv_file_list

output_filename = os.path.join(path, 'output.hdf')

for csv_file_name in csv_file_list:
    print(csv_file_name)
    chunk_container = pd.read_csv(csv_file_name, chunksize=chunksize)
    for chunk in chunk_container:
        chunk.to_hdf(output_filename, key='df', mode="a", index=False)

print('Finished')
