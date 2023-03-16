# Forecasting moss & lichen fractional cover 
## with a Neural Network using Keras
## (Reading X and Y train/test files stored locally)
#
# For lichen output only

print('Starting imports')
import numpy as np
import os
import pandas as pd
import seaborn  as sns
from tensorflow import keras
from numpy.random import seed
from tensorflow.random import set_seed
print('Finished imports')

Hlayers = 3
Olayers = 1
epochs = 10
Hactivation = 'relu'
Oactivation='softmax'
print('Hidden layers = ', Hlayers)
print('Output layers = ', Olayers)
print('Hidden layer activation = ', Hactivation)
print('Output layer activation = ', Oactivation)
print('Epochs = ', epochs)
expname = 'TPS_once_' + str(Hlayers) + 'Hlayers-' + str(Olayers) + 'Olayers-' + str(epochs) + 'epochs-' + Hactivation + '-' + Oactivation
print('Experiment name = ', expname)

# Define set two random seeds, one for numpy and one for tensorflow
print('Random seeds')
seed(1)
set_seed(2)

# Read local .hdf file
path = '/opt/uio/data/'

print('Reading X_train')
X_train_file = os.path.join(path, 'X_tps_once_train.hdf')
X_train = pd.read_hdf(X_train_file)

print('Reading X_test')
X_test_file = os.path.join(path, 'X_tps_once_test.hdf')
X_test = pd.read_hdf(X_test_file)

print('Reading y_train')
y_train_file = os.path.join(path, 'y_lichen_once_train.hdf')
y_train = pd.read_hdf(y_train_file)

print('Reading y_test')
y_test_file = os.path.join(path, 'y_lichen_once_test.hdf')
y_test = pd.read_hdf(y_test_file)

# Instantiate a keras.Input class and tell it how big our input is.
print('Instantiating the Keras input class')
inputs = keras.Input(shape = X_train.shape[1])

# Create the hidden layer
print('Creating the hidden layer')
hidden_layer = keras.layers.Dense(Hlayers, activation=Hactivation)(inputs)

# Create the output layer
print('Creating the output layer')
output_layer = keras.layers.Dense(Olayers, activation=Oactivation)(hidden_layer)

# Create a Keras model
print('Creating the model')
model = keras.Model(inputs=inputs, outputs=output_layer)
model.summary()

keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model with a Mean Square Error loss function and the Adam optimizer
print('Compiling the model')
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

# Train the model
print('Training the model')
history = model.fit(X_train, y_train, epochs=epochs)

# Plot the loss history
print('Generating the history loss plot')
lineplot = sns.lineplot(x=history.epoch, y=history.history['loss'])
fig = lineplot.get_figure()
loss_fig_file = os.path.join(path, 'outputs/' + expname + '_loss.png')
fig.savefig(loss_fig_file) 

model_file = os.path.join(path, 'outputs/' + expname + '.h5')
model.save(model_file)

print('Finished!')
