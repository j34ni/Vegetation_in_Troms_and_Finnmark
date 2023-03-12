# Forecasting moss & lichen fractional cover 
## with a Neural Network using Keras
## (Reading input and output files stored locally as .csv)
## (Writing X_train, X_test, y_train, y_test locally as .csv too)
#
# Using t2m only
# For lichen output only

print('Starting imports')
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
print('Finished imports')

# Reading local .csv files
path = '/opt/uio/data/'
input_file = os.path.join(path, 'input_t.csv')
print('Reading ', input_file)
input = pd.read_csv(input_file)
print('Read')

print('Input')
print(f'Number of rows: {input.shape[0]:,}')
print(f'Number of columns: {input.shape[1]}')

output_file = os.path.join(path, 'output.csv')
print('Reading ', output_file)
output = pd.read_csv(output_file)
print('Read')

output = output.drop(columns=['Bare', 'Grass', 'Shrub', 'Tree'])
output.rename(columns = {'Lichen': 'New_lichen'}, inplace = True)

print('Output')
print(f'Number of rows: {output.shape[0]:,}')
print(f'Number of columns: {output.shape[1]}')

# Split data into training and test sets, 
X_train, X_test, y_train, y_test = train_test_split(input, output, test_size = 0.2, random_state = 0, shuffle = True)

print('X_train')
X_train
X_train_file = os.path.join(path, 'X_t_train.hdf')
X_train.to_hdf(X_train_file, key='df', mode="w", index=False)

print('X_test')
X_test
X_test_file = os.path.join(path, 'X_t_test.hdf')
X_test.to_hdf(X_test_file, key='df', mode="w",  index=False)

print('y_train')
y_train
y_train_file = os.path.join(path, 'y_lichen_train.hdf')
y_train.to_hdf(y_train_file, key='dg', mode="w", index=False)

print('y_test')
y_test 
y_test_file = os.path.join(path, 'y_lichen_test.hdf')
y_test.to_hdf(y_test_file, key='dg', mode="w", index=False)

print('Finishedi!')
