This repository contains the code that was used for modifying data, training models, and comparing the different models on different performance metrics.

# Models
The models implemented are reduced versions of the models *Unet*, *TransUNet*, *DeepLabV3+*, and *DenseTrans*. They were reduced in size to be trainable on the limited hardware of a NVIDIA RTX4060 GPU, with 32 GB of RAM.

# Data
The data is from Actueel Hoogtebestand Nederland, specifically the data set AHN4. 
The data is not included in this repository due to the scale. During the conversion from the original DTM files to four-layer files consisting of DTM, Hillshade, Slope and SLRM the combined file size exceeded 3 TB.
See https://viewer.ahn.nl/AHN4/DTM/0/6.5465/52.26738/3 for the original data.

# Conclusions
By comparing the four customized models trained on all unique unordered combinations of the four input layers (DTM, Hillshade, Slope, SLRM), except the empty set, *DeepLabV3+* was found to be best suited for the application.
This study was limited by a small dataset to train and test models on, simplified models due to limited GPU resources, the use of identical hyperparameters for all models, and a limited set of architectures and input channels considered. 
