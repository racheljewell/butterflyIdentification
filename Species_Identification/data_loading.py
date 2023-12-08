import os
import sys
import numpy as np
import cv2 as cv

def load_testing_data():
    """
    Reads file names and labels from the "./train/" directory
    Note: You must have the unzipped dataset containing the images and csv file in this directory for this function to work.
    Returns:
        labels: an np array containing the butterfly species labels as integers
        images: a list of the images; for this data set images are (224,224,3)
        images_segmented: a list of the segmented images, shape (224,224,3), with the backgroudn removed from the image
        label_dict: a dictionary that maps the integer value for the butterfly specie to the string
    """

    # Get the absolute path of the script's directory
    current_directory = os.path.abspath(os.path.dirname(__file__))
    # Get the parent directory
    parent_directory = os.path.dirname(current_directory)
    # Add the parent directory to sys.path
    sys.path.append(parent_directory)

    data_file_path = os.path.join(current_directory, 'train', 'Training_set.csv')
    

    ################# Load the data #################
    # Load the data. Use ',' as the delimiter
    data = np.loadtxt(data_file_path,dtype=str,  delimiter=',')
    img_file_names = data[1:, 0]
    labels_str = data[1:, 1:]

    # print(labels_str[0,0])

    label_dict = {}
    labels = []
    index = 0

    for label in labels_str:
        if label[0] not in label_dict:
            label_dict[label[0]] = index
            index += 1
        labels.append([label_dict[label[0]]])


    images = []
    images_segmented = []
    for img_file_name in img_file_names:
        img_file_path = os.path.join(current_directory, 'train', img_file_name)
        image = cv.imread(img_file_path)
        images.append(image)
        images_segmented.append(segmentation(image))

    # Following code is for use if image segmentation has been performed in the data_preprocessing jupyter notebook
    # folder = os.path.join(current_directory, 'segmented')
    
    # # If images have been previuosly se
    # for i in range(6499):
    #     img_name = "img_" + str(i) + ".jpg"
    #     img_file_path = os.path.join(folder, img_name)
    #     image = cv.imread(img_file_path)
    #     images_segmented.append(image)
            
    return labels, images, images_segmented, label_dict

"""
    A function that can be used to save segmented images.
"""
def save_data(images):
    # Get the absolute path of the script's directory
    current_directory = os.path.abspath(os.path.dirname(__file__))
    # Get the parent directory
    parent_directory = os.path.dirname(current_directory)
    # Add the parent directory to sys.path
    sys.path.append(parent_directory)

    data_file_path = os.path.join(current_directory, 'segmented')
    count = 0
    for image in images:
        file_name = "img_" + str(count) + ".jpg"
        
        img_file_path = os.path.join(data_file_path, file_name)
        count += 1
        # return img_file_path
        val = cv.imwrite(img_file_path, image)
        
    return count

"""
    Gets the integer key associate with a butterfly species from the dictionary.
"""
def get_key(val, dict):
   
    for key, value in dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"

"""
    Gets the string butterfly species value from the integer key
"""
def get_val(k, dict):

    for key, value in dict.items():
        if k == key:
            return value
        
    return "value doesn't exist"

"""
    Title: Butterfly - GrabCut + DenseNet(CNN)
    Author: Pacawat Panjabud
    Date: 8/2023
    Code version: 5.0
    Availability: https://www.kaggle.com/code/pacawat/butterfly-grabcut-densenet-cnn
    Description: This code removes the backfrom from the butterfly images
"""
def segmentation( RGB_image ):
    mask = np.zeros(RGB_image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    size = RGB_image.shape[0]
    rect = (size//10,size//10,size-size//7,size-size//7)
    cv.grabCut(RGB_image,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = RGB_image*mask2[:,:,np.newaxis]
    
    # Segmentation Failed because the butterfly blended in too much with the background.
    if np.sum(mask2)/(mask2.shape[0]*mask2.shape[1]) < 0.1 :
        return RGB_image
    
    else :
        return img
