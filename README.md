# Vegetation_in_Troms_and_Finnmark
Evaluate potential correlations between the occurrence of extreme winter warming and some types of Arctic vegetation dying (in particular mosses and lichens).

This is what the various Jupyter notebooks do:

**prob_mean_tp1n.ipynb** the inputs and outputs for the ML algorithm :
 * taking Copernicus World Land Cover data at 100m x 100m resolution from 2015 to 2019
 * identifying each year the locations with moss & lichen
 * extracting the corresponding ERA5-Land 2m temperature (t2m) and total precipitation (tp)
 * also finding WLC data for the following year

and produces .hdf files with x_year and y_year);

**merge_mean_tp1n.ipynb** combines the yearly .hdf files into input & output .csv files;

**deep_mean_tp1n.ipynb** reads the input and output .csv files and split them into X_train, X_test, y_train and y_test (80% for training and 20% for testing, randomly shuffled);

**train_mooc_tp1n.ipynb**  
 * instantiates a keras.Input class
 * defines the hidden layer and the corresponding activation function
 * creates the output layer with the output activation
 * creates a Keras model
 * compile the model with a Huber loss function and the Adadelta optimizer
 * trains the model a number of epocs
 * plots the loss history
 * performs a forecast.

**CGLC_download.ipynb**
 * downloads Coperncus Global Land Cover data (100x100m horizontal resolution)
 
**ERA5-land_download.ipynb**
 * downloads ERA5-land data hourly, from 2015 

(Note that writing to CESNET' s3 requires credentials bosdisclosed i the notebooks)
