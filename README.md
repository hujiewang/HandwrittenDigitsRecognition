HandwrittenDigitsRecognition
============================

Implementation of a Neural Network with three layers for handwritten digits recognition.

The neural network has 3 layers - an input layer, a hidden layer and an output layer.
Images are pixel values of digit images and of size 20 x 20, which gives 400 input layer units.
The cost function is computed by using Feedforward and gradiant is computed by doing a Backpropagation.

Note that the cost function is also regularized.

This project is part of Stanford Machine Learning course on Coursera.
  
## Tutorial

  1.Starting the Octave and move to the folder which contains all the source files.
  
  2.Type
  
    
    training
    
    
  in the Octave to train a Neural Network. It also saves weights(Theta_1, Theta_2) into wieghts.mat which will be used by prediction.m .
  
  3.Type
  
    
    prediction
    
    
  in the Octave to predict a random picture of a digit.
