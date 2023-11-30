import os
import sys
import numpy as np
import cv2 as cv

def load_testing_data():
    """
    Reads file names and labels from "train/Training_set.csv.
    Returns:
        labels: an np array containing the butterfly species labels
        images: a list of the images; for this data set images are (224,224,3)
    """

    # Get the absolute path of the script's directory
    current_directory = os.path.abspath(os.path.dirname(__file__))
    # Get the parent directory
    parent_directory = os.path.dirname(current_directory)
    # Add the parent directory to sys.path
    sys.path.append(parent_directory)

    data_file_path = os.path.join(current_directory, 'train', 'Training_set.csv')
    output_dir = os.path.join(current_directory, 'output')

    ################# Load the data #################
    # Load the data. Use ',' as the delimiter
    data = np.loadtxt(data_file_path,dtype=str,  delimiter=',')
    img_file_names = data[1:, 0]
    labels = data[1:, 1:]


    images = []
    for img_file_name in img_file_names:
        img_file_path = os.path.join(current_directory, 'train', img_file_name)
        image = cv.imread(img_file_path)
        images.append(image)

    
    return labels, images
