# Vegetation_in_Troms_and_Finnmark
Evaluate potential correlations between the occurrence of extreme events and some types of Arctic vegetation dying (in particular mosses and lichens).

This is what the various Python scripts do:

**prepare.py** the inputs and outputs for the ML algorithm :
 * taking Copernicus World Land Cover data at 100m x 100m resolution from 2015 to 2019
 * identifying each year the locations with moss & lichen
 * extracting the corresponding ERA5-Land 2m temperature (t2m), total precipitation (tp) and snow deth (sd)
 * also finding WLC data for the following year

and produces .hdf files with x_year and y_year);

**merge.py** combines the yearly .hdf files into input & output .csv files;

**deep.py** reads the input and output .csv files and split them into X_train, X_test, y_train and y_test (80% for training and 20% for testing, randomly shuffled);

**train.py**  
 * instantiates a keras.Input class
 * defines the hidden layer and the corresponding activation function
 * creates the output layer with the output activation
 * creates a Keras model
 * compile the model with a Categorical Crossentropy loss function and the Adam optimizer
 * trains the model a number of epocs
 * plots the loss history
 * performs a forecast.
