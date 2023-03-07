# Forecasting moss & lichen fractional cover 
## with a Neural Network using Keras
## (Reading X and Y train/test files stored locally)

print('Starting imports')
import numpy as np
import os
import pandas as pd
from tensorflow import keras
from numpy.random import seed
from tensorflow.random import set_seed
print('Finished imports')

# Define set two random seeds, one for numpy and one for tensorflow
print('Randon seeds')
seed(1)
set_seed(2)

# Read local .hdf file
path = '/opt/uio/data/'

print('Reading Xn_train')
Xn_train_file = os.path.join(path, 'Xn_train.hdf')
Xn_train = pd.read_hdf(Xn_train_file)

print('Reading Xn_test')
Xn_test_file = os.path.join(path, 'Xn_test.hdf')
Xn_test = pd.read_hdf(Xn_test_file)

print('Reading yn_train')
yn_train_file = os.path.join(path, 'yn_train.hdf')
yn_train = pd.read_hdf(yn_train_file)

print('Reading yn_test')
yn_test_file = os.path.join(path, 'yn_test.hdf')
yn_test = pd.read_hdf(yn_test_file)

# Instantiate a keras.Input class and tell it how big our input is.
print('Instantiating the Keras input class')
inputs = keras.Input(shape = Xn_train.shape[1])

# Create the hidden layer
print('Creating the hidden layer')
hidden_layer = keras.layers.Dense(100, activation="relu")(inputs)

# Create the output layer
print('Creating the output layer')
output_layer = keras.layers.Dense(5, activation="softmax")(hidden_layer)

# Create a Keras model
print('Creating the model')
model = keras.Model(inputs=inputs, outputs=output_layer)
model.summary()

# Compile the model with a Categorical Crossentropy loss function and the Adam optimizer
print('Compiling the model')
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())

# Train the model
print('Training the model')
history = model.fit(Xn_train, yn_train, epochs=100)

# Plot the loss history
print('Generating the history loss plot')
lineplot = sns.lineplot(x=history.epoch, y=history.history['loss'])
fig = lineplot.get_figure()
fig.savefig("loss.png") 

# Perform a prediction
print('Forecasting')
yn_forecast = model.predict(Xn_test)
forecast = pd.DataFrame(yn_forecast, columns=target.columns)
forecast

print('Finished')
