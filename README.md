# Face Detection Using Edge Base

- In construction

**Gabriel Mattheus Bezerra Alves de Carvalho and  Wallace Cruz de Souza**.

## Project

This project intends to identify human faces in images using techniques of image processing, such as image segmentation and image enhancement. The project application is computational photography, applying the results in autofocus in cameras.
Given image, return an image with human face delimited by a square.

### Steps

1\: Remove noise using median filter

2\: Contrast adjustment using histogram equalization

3\: Construct the edge image applying sobel operator

4\: Segment image into blocks

5\: Evaluate features of the blocks

6\: Use features values into a Backpropagation Neural Network (BPN) to classify the block as face or non-face
