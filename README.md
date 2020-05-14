# flower - classification

Flower classification from images using the [Flower Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition) dataset!


# Description

A convolutional neural network is used to recognize the type of flower from an image. A Data Generator is used for the augmentation of the dataset to achieve higher accuracy. There are 5 types of flowers: daisy, dandelion, rose, sunflower and tulip. 

## File Organization

1. [open.py](https://github.com/chrigkou/flower-classification/blob/master/open.py#L37): storing of the images, noise removal using GaussianBlur and preprocessing of the images and the labels
2. [cnn.py](https://github.com/chrigkou/flower-classification/blob/master/cnn.py): image generation using ImageDataGenerator, cnn definition, training and evaluation

## Run experiment

To train and evaluate the cnn classifier all you have to do is run cnn.py. 




