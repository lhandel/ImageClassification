# ImageClassification
In this paper, three different convolutional neural network, AlexNet, VGG-Y and VGG-F has been implemented to perform image classification of the Tiny ImageNet dataset. This was done as a final project in the course DD2424 Deep Learning in Data Science at KTH 2018.

# Data
The Tiny ImageNet dataset consists of the same data as the original ImageNet but the images are cropped into size of 64x64 from 224x224. It has 200 classes and 500 training images for each of the classes, resulting in a training data of 100 000 images [8]. In addition to the training data, it has 10,000 validation images and 10,000 test images (50 for each class). It uses one-hot labels. Some ambiguities in the dataset is caused since it is down-sampled from 224x224 to 64x64. The effect of this down-sampling includes loss of details which makes it harder to locating small objects.

# Implementation
The models were implemented using the open-source Google deep learning framework Tensorflow, stacked with the higher level API, Keras. Keras is designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible.

Data augmentation was implemented using Keras built in ImageGenerator [18] for all models while training. ImageGenerator generates batches of tensor images that are rescales, zoomed-in and flipped-version of the original images. This was done to help train a more accurate and general model.


# Final Architectures
![Final Architectures](https://user-images.githubusercontent.com/13455815/44313625-fd88da00-a40b-11e8-9f47-7f09bb0a1014.png)

# Results
| Model               | Training Accuracy          | Validation Accuracy|
| --------------------| ------------- | --------------------------------|
| AlexNet             | 0.4673        | 0.3292               |
| VGG-Y               | 0.3986        | 0.3225               |
| VGG-F               | 0.4042        | 0.3141               |
