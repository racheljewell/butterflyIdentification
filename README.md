# butterflyIdentification

### By: Rachel Jewell & Sarah Davidson

## File list

This repository contains the following files and folders:

- `README.md`: This file.
- Species_Identification: This folder contains files pertaining to the butterfly species identification portion of this project.
    - `requirements.txt`: This file contains the list of Python packages required to run the code in this folder
    - `data_loading.py`: This file contains functionality for loading and working with the dataset for this project
    - `data_preprossessing.ipynb`: This file is a Jupyter Notebook containing details about and the preprocessing of the butterfly dataset
    - `Species_CNN.ipynb`: This file is a Jupyter Notebook containing the source code where the data is split and the CNN is built, trained, and evaluated
    - `cnn_models`: a folder that holds saved models
        - `finalModel`: the final, and most accurate model
- detect_butterfly_model: This Juppter Notebook containes the source code and tests computing butterfly detection
- Final Reports: This repo contains 2 final reports
      - `Species_Identification_Report.pdf`: This report is done by Rachel detailing her work on species classification
      - `DetectButterfliesReport.pdf`: This report is done by Sarah detailing her work on butterfly detection
              -NOTE: The report reads in a dataset called "notbtterfly" and the dataset on main is named "notbutterfly" but they are the same dataset. Sarah had to delete the
                      dataset and re-upload it after dealing with a bug
  
      
  

## Setting up the environment

This application should be able to run using the "Computer Vision" environment provided for this course. However, if packages are missing, install the packages listed in the `requirements.txt` file using the following command:


```bash
pip install -r requirements.txt
```

**Note:** If on Github Codespaces after installing the required packages you get the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`, you will need to run the following command in the terminal:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
```

## Required datasets

The required datasets can be found and downloaded at the locations listed below:

    Title: Butterfly Image Classification
    Author: Depie
    Date: 7/2023
    Code version: 1.0
    Availability: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

    Title: Random Image Sample Dataset
    Author: Pankaj Kumar
    Date: 12/2022
    Code version: 1.0
    Availability: https://www.kaggle.com/datasets/pankajkumar2002/random-image-sample-dataset/

## Running the code

The code can be run by following the steps in the Jupyter Notebooks.
