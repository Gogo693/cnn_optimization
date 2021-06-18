# cnn_optimization

## Summary

This project aims at experimenting different optimization techniques on a Classification model based on CNNs using TensorFlow.

## Model & Dataset
The model we use is a simple CNN architecture with the aim to classify between different fashion garments in the MNIST Fashion dataset.

## Optimization Solutions

* REDUCE MODEL SIZE
* PRUNING
* QUANTIZATION
* CLUSTERING
* DISTILLATION

## Files
- main.ipynb: main code and detailed description
- final_model.h5: tf optimized model
- final_model_cml: CoreML optimized model
- run_model.py: script to use a single image for inference using CML model (for MacOS evironment)
- fashion1.png: simple image as input to run_model.py
