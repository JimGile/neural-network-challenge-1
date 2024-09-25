DU-VIRT-AI-PT-05-2024-U-LOLC-MWTH - Module 18 - Neural Network Challenge 1 (Sep 24, 2024)

# Module 18 Challenge - Neural Network Recomendation Systems

[notebooks/student_loans_with_deep_learning.ipynb](https://github.com/JimGile/neural-network-challenge-1/blob/main/notebooks/student_loans_with_deep_learning.ipynb)

## Focus

Module 18 focuses on part 1 of neural networks and deep learning. It starts with the basics of evaluating neural models using TensorFlow and moves on to exploring deep learning for real-world classification tasks, and recommendation systems using Restricted Boltzmann Machines.

## Day 1 - Introduction to Neural Networks

Neural networks are a powerful machine learning technique modeled after neurons in the brain. They leverage multiple layers of interconnected nodes (or "neurons") which
perform individual computations to model complex patterns. They are used for tasks such as recomendationn systems, image classification, speech recognition, and natural language processing.

* Advantages and disadvantages of using neural network models with other types of machine learning models.
  * NN models can achieve higher accuracy on complex problems but come with higher computational cost and lower interpretability.
  * They are also prone to overfitting of the training data.

* Activation Functions
  * **ReLU**: Rectified Linear Unit. It is the most commonly used activation function.
  * **Sigmoid**: The sigmoid function is a common activation function. It is used in neural networks for classification problems.
  * **Tanh**: The tanh function transforms the output to a range between –1 and 1. The output for a model using a tanh function also forms a characteristic S-curve. It’s primary use is classifying data into one of two classes.
  * **Softmax**: The softmax function is used in neural networks for multi-class classification problems.
  * **Softplus**: The softplus function is used in neural networks for regression problems.

## Day 2 - Neural Networks for Classification

* Deep (multi layer) neural network models using TensorFlow.
* Explore how different neural network structures change algorithm performance.
* Use KerasTuner to assist with finding optimal neural network structures.
* Save trained TensorFlow models for later use.

## Day 3 - Recommendation Systems

Neural network and deep learning models that can be used to develop recommendation systems, either
as a component of the recommendation system, or the whole recommendation system model, include:

* Multilayer perceptrons (MLPs)
* Recurrent neural networks (RNNs)
* Restricted Boltzmann Machines (RBMs)
* Meta’s Deep Learning Recommender Model (DLRM)
* Google's Wide & Deep Neural Network (WDNN)
* Neural Collaborative Filtering (NCF)
* Variational AutoEncoders

## Challenge

The challenge is to build a neural network model to predict whether or not student loans will be repaid, and discuss building a recommendation system for student loan options.

## Solution

The solution is in the Jupyter Notebook file [notebooks/student_loans_with_deep_learning.ipynb](https://github.com/JimGile/neural-network-challenge-1/blob/main/notebooks/student_loans_with_deep_learning.ipynb).
