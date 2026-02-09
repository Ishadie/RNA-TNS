# RNA-TNS: A novel unified Tolerance-based Clustering and Representation Learning framework for m⁶A Site Prediction 
[![Code](https://img.shields.io/badge/Code-Ishadie/RNA--TNS-blue?logo=GitHub)](https://github.com/Ishadie/RNA-TNS)

# Introduction
Over the past few decades, extreme weather events have become more common and violent due to the retreat of the Arctic Sea ice, which has changed regional and global climate patterns. The satellite image patterns show that enormous sea ice loss is critical to Arctic amplification. The Arctic sea ice retreat due to climate change threatens the environment significantly. As a critical component that helps understand the climate crisis, changes in Arctic ice require accurate analysis and prediction. Various researchers have used machine learning and deep learning models for sea ice forecasting. 

This research focuses on processing satellite images using digital image processing techniques and uses a deep learning model that takes multimodal data to analyze time series forecasting of future ice extent. We leverage image processing techniques such as Optical Character Recognition (OCR) for detecting the text of the image and handling missing data, Oriented FAST and Rotated BRIEF (ORB) for aligning images, low-pass filter for denoising satellite images, and Otsu’s thresholding for segmenting ice regions from land and ocean. We then use Canny Edge Detection for feature extraction to highlight sea ice boundaries. We extract contours, calculate the ice retreat percentage from the image, and finally, find changes in ice coverage using image subtraction. After processing the images, we use a transformer-based model to perform a time series prediction of future ice extent. The transformer-based model is designed in a way that it takes multimodal data, one modality is hand-crafted numerical features from satellite images, while the other modality is processed satellite images. 

![DIP PROPOSAL-Copy of dip final project drawio](https://github.com/user-attachments/assets/66313918-7057-44cd-991d-eb22eeb2044f)


# Dataset
We use Arctic sea ice data acquired from the National Snow and Ice Data Center (NSIDC) in this research. The data we are using consists of images from July 16th to August 16th of the years 1979 to 2024. If there is no data at all for a day, the image is labeled with “NO DATA”. The naming convention ```“h_yyyymmdd_extn_[blmrbl]_[hires]_vX.x.png”``` is used for labeling the images.

The whole dataset is available ​​at https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/images/

Another dataset we use is the Sea Surface Temperature (SST) data corresponding to the Arctic sea ice image data. We leverage the weekly mean SST data for our experiments. The data is available in degrees Celsius.

The SST dataset can be accessed from https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html

# Digital Image Processing Techniques
## Pre-Processing Polar Image Data

Data is pre-processed using the following image processing techniques:
- ```Optical Character Recognition (OCR)```: The missing data are handled using OCR to detect the specific text(NO DATA) from images and remove such images.
  
![no data](https://github.com/user-attachments/assets/82bafdbd-155d-40c3-84fe-dd7b4dd3acd7)

- ```Resizing```: All images are resized maintaining a uniform size (512x512) for alignment.
- ```Oriented FAST and Rotated BRIEF (ORB)```: Images are aligned to a reference image using keypoint matching and geometric transformation, which is used for correcting shifts or rotations between different temporal images of the same region.
- ```Gaussian Blur```: Gaussian Blur is used to reduce sensor noise and smooth atmospheric distortions.
- ```Grayscale Conversion & Normalization```: Gray scale conversion is performed to reduce computational complexity, and normalization ensures uniform contrast across all images.
- ```Otsu's Thresholding```: Otsu's thresholding is used to automatically separate sea ice (white) from background (black) using intensity distribution.


![preprocess](https://github.com/user-attachments/assets/dce19c1a-1f04-412a-ac76-7fb1a03f5615)


## Feature Extraction from Polar Image Data

From the pre-processed images, three handcrafted features - ice coverage, ice retreat percentage from the previous year, and ice retreat percentage since 1979 are extracted by using the following techniques:

- ```Circular Masking```: We have applied a circular binary mask to remove legends, borders and kept only the polar region.
- ```Edge Detection```: Canny Edge detection has been used to highlight the sharp boundaries of sea ice edges.
- ```Contour Detection```: Contours are extracted from the ice mask.
- ```Ice Area Calculation```: We have measured the pixel area of all valid contours and computed the ice coverage as a percentage of the total image size.
- ```Image Subtraction```: Image subtraction is performed to find the absolute pixel difference between two masks (previous vs. current year).
- ```Retreated Area Calculation```: Retreat area is computed as a percentage of total image area, based on the thresholded difference mask.
- ```Overlay Visualization```: Overlay visualization has been performed to evaluate if the ice-covered area is properly detected.

  ![image feature extraction](https://github.com/user-attachments/assets/b96bfb44-f209-4792-9d88-8e5035589a3a)

# Steps to Reproduce The Result

## Requirements

- **Python** 3.8–3.11  
- All other dependencies are listed in [requirements.txt](https://github.com/Ishadie/DIP_Predicting_SIE/blob/68ed61ca821070c29f7458a27990214f5ecb64b1/requirements.txt)
- You can install them using:
````
pip install -r requirements.txt
````
- To install jupyter to your system run the following commands:
````
pip install notebook
jupyter notebook
python -m notebook
````
- Please make sure you are in the root directory of the project when running the Jupyter commands
- Use the same Python version and dependencies as listed to avoid runtime issues.
## Directory Structure Overview
<pre> <code> 
   
  ├── sample_image/                                    # Contains example satellite images 
  ├── results/                                         # Contains the model prediction outputs 
  ├── logs/                                            # Optuna logs and final model logs 
  ├── featureExtraction.ipynb                          # End-to-end image feature extraction 
  ├── imageProcessingSteps.ipynb                       # Visualize and understand image preprocessing 
  ├── siePredictionRNN.ipynb                           # RNN model (unimodal - numerical) 
  ├── sieNumericalTransformerGRUModel.ipynb                 # Transformer+GRU model (unimodal - numerical) 
  ├── siePredictionMultimodalModel.ipynb      # Main model (Transformer+GRU multimodal) 
  ├── ice_extent_full_dataset.csv                      # Final dataset with extracted features 
  ├── requirements.txt  </code> </pre> 

## Training and Evaluation
Below is the description of each file and how the results can be reproduced.

- To visualize and analyze how each image is processed and how the features are extracted, download and run this file [imageProcessingSteps.ipynb](https://github.com/Ishadie/DIP_Predicting_SIE/blob/824f892fd395815935b731df3e0eec26ac37588e/image_processing/imageProcessingSteps.ipynb)
In your terminal or command prompt, from the directory where the notebook file is located, run:
````
jupyter notebook imageProcessingSteps.ipynb

````
Also, the images that are used for analyzing can be found in [sample_image](https://github.com/Ishadie/DIP_Predicting_SIE/tree/0b855763649b942985a759f5e363e46ad339db53/sample_image)
- For downloading Arctic Sea Ice data through the automated script and to pre-process and extract all the features from the image data, download and run this file [featureExtraction.ipynb](https://github.com/Ishadie/DIP_Predicting_SIE/blob/63cb10f94cbc99b2d7d5a4d0d20766033c4fb2ae/image_processing/featureExtraction.ipynb). In your terminal or command prompt, from the directory where the notebook file is located, run:
````
jupyter notebook featureExtraction.ipynb

````
  
  ```Note:``` This step can be skipped to reproduce the result, as all the images are processed for years 1979-2024. The final dataset with extracted features and the sst data can be found and used by downloading this CSV : [ice_extent_with_sst.csv](https://github.com/Ishadie/DIP_Predicting_SIE/blob/dbceeadbe6fa6c98ca390eb307fd832ff3b46697/ice_extent_with_sst.csv)) 

- For the unimodal RNN model on numerical features, download and run this file [siePredictionRNN.ipynb](https://github.com/Ishadie/DIP_Predicting_SIE/blob/63cb10f94cbc99b2d7d5a4d0d20766033c4fb2ae/unimodal_approach/siePredictionRNN.ipynb). In your terminal or command prompt, from the directory where the notebook file is located, run:
````
jupyter notebook siePredictionRNN.ipynb

````
- For the unimodal Transformer-GRU model on numerical features, download and run this file [sieNumericalTransformerGRUModel.ipynb]
(https://github.com/Ishadie/DIP_Predicting_SIE/blob/82280d343977feaefd95cfbeeef4f1c0ebcb6b4c/unimodal_approach/sieNumericalTransformerGRUModel.ipynb). In your terminal or command prompt, from the directory where the notebook file is located, run:
````
jupyter notebook sieNumericalTransformerGRUModel.ipynb

````
- Our main model, multimodal Transformer-GRU for Sea-Ice Extent Forecasting, can be found in [siePredictionMultimodalModel.ipynb](https://github.com/Ishadie/DIP_Predicting_SIE/blob/82280d343977feaefd95cfbeeef4f1c0ebcb6b4c/multimodal_approach/siePredictionMultimodalModel.ipynb). Download and run this to reproduce the results. To run the file, in your terminal or command prompt, from the directory where the notebook file is located, run:
````
jupyter notebook siePredictionMultimodalModel.ipynb

````
- The results we got from our experiment can be found in the folder: [results](https://github.com/Ishadie/DIP_Predicting_SIE/tree/63cb10f94cbc99b2d7d5a4d0d20766033c4fb2ae/results)







