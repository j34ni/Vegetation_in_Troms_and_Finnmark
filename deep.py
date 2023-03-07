# Forecasting moss & lichen fractional cover 
## with a Neural Network using Keras
## (Reading input and output files stored locally)

print('Starting imports')
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
print('Finished imports')

# Reading local .hdf files
print('Read input')
path = '/opt/uio/data/'
input_file = os.path.join(path, 'input.hdf')
input = pd.read_hdf(input_file)

print('Input')
print(f'Number of rows: {input.shape[0]:,}')
print(f'Number of columns: {input.shape[1]}')

print('Read output')
output_file = os.path.join(path, 'output.hdf')
output = pd.read_hdf(output_file)

output.rename(columns = {'Bare': 'New_bare', 'Grass': 'New_grass', 'Lichen': 'New_lichen', 'Shrub': 'New_shrub', 'Tree': 'New_tree'}, inplace = True)

print('Output')
print(f'Number of rows: {output.shape[0]:,}')
print(f'Number of columns: {output.shape[1]}')

# Split data into training and test sets, 
Xn_train, Xn_test, yn_train, yn_test = train_test_split(input, output, test_size = 0.2, random_state = 0, shuffle = True)

print('Xn_train')
Xn_train
Xn_train_file = os.path.join(path, 'Xn_train.hdf')
Xn_train.to_hdf(Xn_train_file, key='df', mode="w", index=False)

print('Xn_test')
Xn_test
Xn_test_file = os.path.join(path, 'Xn_test.hdf')
Xn_test.to_hdf(Xn_test_file, key='df', mode="w",  index=False)

print('yn_train')
yn_train
yn_train_file = os.path.join(path, 'yn_train.hdf')
yn_train.to_hdf(yn_train_file, key='df', mode="w", index=False)

print('yn_test')
yn_test 
yn_test_file = os.path.join(path, 'yn_test.hdf')
yn_test.to_hdf(yn_test_file, key='df', mode="w", index=False)

print('Finishedi!')
