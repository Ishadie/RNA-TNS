# RNA-TNS: A novel unified Tolerance-based Clustering and Representation Learning framework for m⁶A Site Prediction 
[![Code](https://img.shields.io/badge/Code-Ishadie/RNA--TNS-blue?logo=GitHub)](https://github.com/Ishadie/RNA-TNS)

# Background: 
N6-methyladenosine (m6A) is the most abundant internal modification in eukaryotic RNA and plays
a critical role in regulating RNA stability, splicing, and translation. Accurate computational identification of m⁶A
modification sites remains challenging due to complex sequence patterns, intra-class variability, and limited cross-
chromosome generalization.
# Methods: 
In this paper, we propose RNA-TNS, a novel unified tolerance near set-based clustering and representation
learning framework for m6A site prediction. The proposed framework integrates tolerance relation-based neighbor-
hood classes with prototype driven supervised learning of fused embeddings. Fused embeddings are derived from
RNA sequences which are encoded using multi resolution feature representations that combine k-mer composition
with embeddings derived from convolutional neural networks and pretrained nucleotide transformer models, includ-
ing DNABERT and the Nucleotide Transformer. Label specific tolerance (neighborhood) classes are clustered using
configurable distance measures and tolerance thresholds. Classification is performed through distance weighted voting
over the nearest tolerance class representative prototypes.
# Results: 
Experimental results evaluation across both random and chromosome-level splits indicates
that RNA-TNS delivers reliable predictive performance. In the Random Split setting, the model improves accuracy
by 3.11%, sensitivity by 0.88%, specificity by 2.44%, Matthews Correlation Coefficient by 2.12%, and AUC-ROC
by 1.06% compared to the previous works. Under the Leave-One-Chromosome-Out independent test setting, RNA-
TNS improves accuracy by 2.09%, sensitivity by 7.46%, specificity by 0.19%, Matthews Correlation Coefficient by
33.82%, and AUC-ROC by 1.20%. These results suggest that the proposed framework remains effective under realistic
distributional shifts. The time complexity of RNA-TNS is O(n2d), where n denotes the number of training samples
and d is the dimensionality of the feature representation. Also, RNA-TNS remains computationally feasible due to
its prototype-based formulation and one time training. Visualization of learned representations with t-SNE further
shows improved class separation at the prototype level.


# Dataset
We use a benchmark m⁶A dataset curated from the m6A-Atlas database, which provides base
resolution annotations of RNA N6-methyladenosine modification sites across the human genome.

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







