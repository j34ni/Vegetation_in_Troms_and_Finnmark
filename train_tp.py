# Forecasting moss & lichen fractional cover 
## with a Neural Network using Keras
## (Reading X and Y train/test files stored locally)
### Using only 2m temperature and total precipitation from ERA5-land
### For lichen output only

print('Starting imports')
import numpy as np
import os
import pandas as pd
import seaborn  as sns
from tensorflow.keras.layers import Input, BatchNormalization, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
print('Finished imports')

print('Defining various parameters')

depth = 32
width = 8192
epochs = 3 
activation = 'tanh'
final_activation = 'linear'
epochs = 3
batch_size = 512
learning_rate = 10**(-3)
loss = 'mse'
validation_split = 0.2
print('Depth = ', depth)
print('Width = ', width)
print('Hidden layer activation = ', activation)
print('Final activation = ', final_activation)
print('Learning rate = ', learning_rate)
print('Epochs = ', epochs)
print('Batch size = ', batch_size)
print('Loss function = ', loss)
print('Validation split = ', validation_split)
expname = 'TP_twice_' + str(depth) + '_' + str(width) + '_' + str(epochs) + 'epochs-' + str(batch_size) + '_' + activation + '-' + final_activation + '-' + str(learning_rate) + '-' + loss + '-' + str(validation_split)
print('Experiment name = ', expname)

# Read local .hdf file
path = '/opt/uio/data/'

input_file = os.path.join(path, 'X_tp_twice_train.hdf')
print('Reading ', input_file)
input_train = pd.read_hdf(input_file)
print('Read')

print('Input')
print(f'Number of rows: {input_train.shape[0]:,}')
print(f'Number of columns: {input_train.shape[1]}')

output_file = os.path.join(path, 'y_lichen_twice_train.hdf')
print('Reading ', output_file)
output_train = pd.read_hdf(output_file)
print('Read')

print('Output')
print(f'Number of rows: {output_train.shape[0]:,}')
print(f'Number of columns: {output_train.shape[1]}')

input_file = os.path.join(path, 'X_tp_twice_test.hdf')
print('Reading ', input_file)
input_eval = pd.read_hdf(input_file)
print('Read')

print('Input')
print(f'Number of rows: {input_eval.shape[0]:,}')
print(f'Number of columns: {input_eval.shape[1]}')

output_file = os.path.join(path, 'y_lichen_twice_test.hdf')
print('Reading ', output_file)
output_eval = pd.read_hdf(output_file)
print('Read')

print('Output')
print(f'Number of rows: {output_eval.shape[0]:,}')
print(f'Number of columns: {output_eval.shape[1]}')

def fullyconnected_sequential(
    input_shape: int,  # How many predictors?
    width: int,  # How wide should the layers be?
    depth: int,  # How many layers?
    activation: str,  # What nonlinearity to use?
    final_activation: str,  # Output layer?
    learning_rate: float,  # What learning rate?
    loss: str,  # What loss function?
):
    # Create a model object
    model = Sequential()

    # Then just stack layers on top of one another
    # the first specifies the shape of the inputs expected input
    model.add(Input(input_shape, name = 'Inputs'))

    # Then we stack on depth number of consectutive dense layers
    # To write more compact code we can include the activation
    # function we want to apply after each Dense block in the
    # call itself.
    for i in range(depth):
        model.add(Dense(width, activation = activation))

    # Finally we add an output layer, we want to predict
    # 1 variable, and we will probably use a linear output
    # layer, so we don't constrain the output
    model.add(Dense(1, activation = final_activation))

    # Next we need to specify the optimiser we want to use and what learning rate to use
    opt = Adam(learning_rate)

    # Finally we compile the model, specifying the loss we want to minimise
    model.compile(loss = loss, optimizer = opt)

    # Afterwards we can summarise the model to see the shapes
    model.summary()
    return model

def fullyconnected_functional(
    input_shape: int,  # How many predictors?
    width: int,  # How wide should the layers be?
    depth: int,  # How many layers?
    activation: str,  # What nonlinearity to use?
    final_activation: str,  # Output layer?
    learning_rate: float,  # What learning rate?
    loss: str,  # What loss function?
):
    # First we create an input layer
    inputs = Input(input_shape, name='Inputs')

    # Pass it to the hidden variable we will reuse
    hidden = inputs

    for i in range(depth):
        # Now we repeatedly apply a Dense layer to the object each time this adds another layer onto the stack
        hidden = Dense(width, activation = activation)(hidden)

    # Finally stitch on the output layer
    output = Dense(1, activation = final_activation)(hidden)

    # And the model itself is created by specifying the input layers and output layers.
    model = Model(inputs = inputs, outputs = output)

    # Next we need to specify the optimiser we want to use and what learning rate to use
    opt = Adam(learning_rate)

    # Finally we compile the model, specifying the loss we want to minimise
    model.compile(loss = loss, optimizer = opt)

    # Afterwards we can summarise the model to see the shapes
    model.summary()
    return model

model = fullyconnected_sequential(input_train.shape[1], depth=depth, width=width, activation=activation, final_activation=final_activation, learning_rate=learning_rate, loss=loss)

# model = fullyconnected_functional(input_train.shape[1], depth = depth, width = width, activation = activation, final_activation = final_activation, learning_rate = learning_rate, loss = loss)

# Sequential
model.fit(input_train, output_train, validation_split = validation_split, batch_size = batch_size, epochs = epochs)

# Functional
# model.fit(input_train, output_train, validation_split = validation_split, batch_size = batch_size, epochs = epochs)

model_file = os.path.join(path, 'outputs/' + expname + '.h5')
print('Saving model in: ', model_file)
model.save(model_file)

# Model evaluation on the training data
print('Model evaluation on the training data')
train_loss = model.evaluate(input_train, output_train)
print('Training Mean Squared Error:', train_loss)

# Model performance on unseen data
print('Model performance on unseen data')
val_loss = model.evaluate(input_eval, output_eval)
print('Validation Mean Squared Error:', val_loss)

print('Finished!')
