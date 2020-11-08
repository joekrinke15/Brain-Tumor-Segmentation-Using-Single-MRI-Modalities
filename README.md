
# Model Architecture
![unet](https://raw.githubusercontent.com/joekrinke15/Brain-Tumor-Segmentation-Using-Single-MRI-Modalities/master/ReadMeImages/unet.png)

# Training Methodology 
MRI segmentation models are usually trained with each modality as an input channel to the model. 

![modalities](https://raw.githubusercontent.com/joekrinke15/Brain-Tumor-Segmentation-Using-Single-MRI-Modalities/master/ReadMeImages/concat.PNG)

Our model, in contrast, trains on individual modality-image pairs. 

![novelinput](https://raw.githubusercontent.com/joekrinke15/Brain-Tumor-Segmentation-Using-Single-MRI-Modalities/master/ReadMeImages/individual.PNG)

# Sample Segmenatations 

![segmentations](https://raw.githubusercontent.com/joekrinke15/Brain-Tumor-Segmentation-Using-Single-MRI-Modalities/master/ReadMeImages/samplesegmentation.png)

# Validation Performance on BraTS 2015 Dataset
| Dice Score | Precision | Recall |
|:----------:|:---------:|:------:|
|    .8422   |   .8330   |  .8608 |

# Citations
