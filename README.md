# README

## 1. Introduction

This is the submission code of the moon\_light team for the IJCNN\_2025 competition. The code is designed to address challenges in graph label noise learning.

## 2. Problem Statement

Two major issues plague graph label noise learning: overfitting to noisy labels and underfitting caused by early stopping. Overfitting to noisy labels leads the model to memorize incorrect information, while underfitting due to early stopping prevents the model from fully learning the underlying patterns in the data.

## 3. Proposed Solution

To tackle these problems, our team has proposed a simple yet effective training mechanism with three key aspects:

**Avoiding Overfitting to Noisy Labels**: We incorporated the dropout mechanism during the feature learning process. Dropout randomly "drops out" (sets to zero) a fraction of the input units in a neural network, which helps prevent the model from relying too much on any single feature and thus reduces the risk of overfitting to noisy labels.

**Preventing Underfitting**: We increased the number of training epochs to 500. By extending the training time, the model has more opportunities to learn the complex relationships in the data, minimizing the impact of underfitting caused by early stopping.

**Adding Virtual Nodes in the Graph Network**: Incorporating virtual nodes in the graph network enriches the graph structure. These virtual nodes can act as additional information carriers, helping the model better capture global and local patterns in the graph data, which is beneficial for learning in the presence of label noise.
