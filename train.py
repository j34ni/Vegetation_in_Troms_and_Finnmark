# Forecasting moss & lichen fractional cover 
## with a Neural Network using Keras
## (Reading X and Y train/test files stored locally)

print('Starting imports')
import numpy as np
import os
import pandas as pd
import seaborn  as sns
from tensorflow import keras
from numpy.random import seed
from tensorflow.random import set_seed
print('Finished imports')

layers = 10
epochs = 100
print('Layers = ', layers)
print('Epochs = ', epochs)

# Define set two random seeds, one for numpy and one for tensorflow
print('Random seeds')
seed(1)
set_seed(2)

# Read local .hdf file
path = '/opt/uio/data/'

print('Reading X_train')
X_train_file = os.path.join(path, 'X_train.hdf')
X_train = pd.read_hdf(X_train_file)

print('Reading X_test')
X_test_file = os.path.join(path, 'X_test.hdf')
X_test = pd.read_hdf(X_test_file)

print('Reading y_train')
y_train_file = os.path.join(path, 'y_train.hdf')
y_train = pd.read_hdf(y_train_file)

print('Reading y_test')
y_test_file = os.path.join(path, 'y_test.hdf')
y_test = pd.read_hdf(y_test_file)

# Instantiate a keras.Input class and tell it how big our input is.
print('Instantiating the Keras input class')
inputs = keras.Input(shape = X_train.shape[1])

# Create the hidden layer
print('Creating the hidden layer')
hidden_layer = keras.layers.Dense(layers, activation="relu")(inputs)

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
history = model.fit(X_train, y_train, epochs=epochs)

# Plot the loss history
print('Generating the history loss plot')
lineplot = sns.lineplot(x=history.epoch, y=history.history['loss'])
fig = lineplot.get_figure()
fig.savefig(str(layers)+'layers-'+str(epochs)+'epochs_loss.png') 

model.save(str(layers)+'layers-'+str(epochs)+'epochs_loss')

# Perform a prediction
print('Forecasting')
y_forecast = model.predict(X_test)
forecast = pd.DataFrame(y_forecast, columns=y_test.columns)
forecast

print('Finished')
