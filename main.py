import os
import sys
import numpy as np

# Get the absolute path of the script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
# Add the parent directory to sys.path
sys.path.append(parent_directory)

data_file_path = os.path.join(current_directory, 'data', 'mnist_data.csv')
output_dir = os.path.join(current_directory, 'output')

################# Load the data #################
# Load the data. Use ',' as the delimiter
data = np.loadtxt(data_file_path, delimiter=',')
labels = data[:, 0]
data = data[:, 1:]
# Reshape the data to be a list of 28x28 2D images

## Need to know the shape of these images
data = data.reshape(data.shape[0], 224, 224)