# Face Detection Using Edge Base

### Authors

- Gabriel Mattheus Bezerra Alves de Carvalho (@GabrielBCarvalho)
- Wallace Cruz de Souza (@Cruwal)

## Abstract

This project intends to identify human faces in images using techniques of image processing, such as image segmentation and image enhancement. The project application is computational photography, applying the results in autofocus in cameras.
Given an image containing one or more faces of people, the same image is returned with squares delimiting the faces contained in it.

## Description

### Steps

1\: Remove noise using median filter;

2\: Contrast adjustment using histogram equalization;

3\: Construct the edge image applying Sobel operator;

4\: Segment image into blocks;

5\: Evaluate features of the blocks;

6\: Use features values into a Backpropagation Neural Network (BPN) to classify the block as face or non-face.

7\: Display the image with squares delimiting the faces

### Input example

The input is represented by an image containing one or more faces. The image can be either grayscale or colored. 
The following is an example of an input image

![alt text](https://files.realpython.com/media/bla2.5577e4ec1f8e.jpg)

### Output example (objective)

The following is an example of the desired output image, based on the previous input image

![alt text](https://files.realpython.com/media/bla3.0a8b11f62c76.jpg)


#### Images references
Traditional Face Detection With Python: https://realpython.com/traditional-face-detection-python/
