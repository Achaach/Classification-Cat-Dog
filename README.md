# Classification-Cat-Dog

## Introduction
In this project we will build a model to do classification using CNN of our image dataset which includes cat and dog images. a convolutional network receives a normal color image as a rectangular box whose width and height are measured by the number of pixels along those dimensions, and whose depth is three layers deep, one for each letter in RGB.As images move through a convolutional network, different patterns are recognised just like a normal neural network. But here rather than focussing on one pixel at a time, a convolutional net takes in square patches of pixels and passes them through a filter. That filter is also a square matrix smaller than the image itself, and equal in size to the patch. It is also called a kernel.

## Data
Dogs vs. Cats: Create an algorithm to distinguish dogs from cats

https://www.kaggle.com/competitions/dogs-vs-cats/data

Data has been splited into training and testing set, containing dogs and cats in each set.

## Model

To buil a CNN model, we mainly follow the steps:

1. Define a Sequential model.

2. Start adding layers to it.

3. Compile the model . 

There are 3 things to successfully buil the model: Loss, Optimizer, Metrics
Loss : To make our model better we either minimize loss or maximize accuracy. NN always minimize loss. To measure it we can use different formulas like 'categorical_crossentropy' or 'binary_crossentropy'. Here I have used binary_crossentropy

Optimizer : If you know a lil bit about mathematics of machine learning you might be familier with local minima or global minima or cost function. To minimize cost function we use different methods For ex :- like gradient descent, stochastic gradient descent. So these are call optimizers. We are using a default one here which is adam

Metrics : This is to denote the measure of your model. Can be accuracy or some other metric.


In this project, we have a input layer to pass the input image. The image will be reshape into 1-D array, for example a 64x64 image will be reshaped to (4096,1) array. Then we have three layers to do the features extraction. Conv Layer will extract features from image and Pooling Layer layerreduce the spatial volume of input image after convolution. Then we have a Fully Connected Layer which connects the network between layers before our Output Layer to give the prediction.

## Result
The Loss&Accuracy vs Epoch are shown below:

![loss](https://user-images.githubusercontent.com/90078254/218219082-bcf6cfac-2997-41f8-9c00-351ea33e147d.png)

<img width="416" alt="accuracy" src="https://user-images.githubusercontent.com/90078254/218219427-35c410f3-3345-4df3-919c-5a704d19ce34.png">

And here is the visualization of the classification on my dataset:

![my_cat_classification](https://user-images.githubusercontent.com/90078254/218219595-be7bfc24-254f-45e1-91c4-515894841527.png)

## Reference:
https://www.kaggle.com/code/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial

https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification

