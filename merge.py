## Merge yearly .hdf files into input & output .csv files

print('Imports')
import glob
import os
import pandas as pd

# Input to .csv
print('Input files')
path = '/opt/uio/data/'
x_filename = 'x_2*.hdf'
file_list = glob.glob(os.path.join(path, x_filename))
file_list.sort()
file_list

input_filename = os.path.join(path, 'input.csv')

for file_name in file_list:
    print(file_name)
    df = pd.read_hdf(file_name)
    df = df.drop(columns=['lat', 'lon'])
    print(df)
    if (file_name == file_list[0]):
         df.to_csv(input_filename, header=True, index=None, sep=',')
    else:
        # Append to file
         df.to_csv(input_filename, mode='a', header=None, index=None, sep=',')


# Output to .csv
print('Output files')
y_filename = 'y_2*.hdf'
file_list = glob.glob(os.path.join(path, y_filename))
file_list.sort()
file_list

output_filename = os.path.join(path, 'output.csv')

for file_name in file_list:
    print(file_name)
    dg = pd.read_hdf(file_name)
    print(dg)
    if (file_name == file_list[0]):
         dg.to_csv(output_filename, header=True, index=None, sep=',')
    else:
        # Append to file
         dg.to_csv(output_filename, mode='a',  header=None, index=None, sep=',')


print('Finished')
