# Authors: Gabriel Mattheus Bezerra Alves de Carvalho and Wallace Cruz de Souza
# SCC0251 - Processamento de Imagens
# 2019, Semester 1
# Final Project: Face Detection  

import numpy as np
import math
import imageio
import matplotlib.pyplot as plt
from skimage import io

def main():
    print("Type the name of the image...")

    # Reading filename and opening the image
    filename = str(input()).rstrip()
    # Original image
    image_input = io.imread(filename)
    # Grayscale image
    image = io.imread(filename, as_gray = True)

    # Apply 2D Median Filter
    image = two_d_median_filter(3, image)

    # Apply Histogram Equalization
    image = histogram_equalization(image)


    # write the result image
    imageio.imwrite("result.jpg", image)
    return True

# 2D Median Filter
def two_d_median_filter(n, image):
    # Sizes of the original image
    size = image.shape

    # Get the centered position
    pivot = int(math.floor(n / 2))

    # Create rows and columns filled with 0 and insert on beginning and end of each axis
    new_image = np.pad(image, ((pivot, pivot), (pivot, pivot)), 'constant', constant_values = (0))

    # Allocate output image
    output_image = np.empty(shape = size, dtype=float)
    
    for i in range(size[0]):
        for j in range(size[1]):
            # Get the necessary submatrix to calculate the value at (i, j)
            sub_matrix = new_image[i: (i + n), j: (j + n)]

            # Calculate the value at (i, j) using the median of the sub_matrix.
            median = np.median(sub_matrix)
            
            output_image[i][j] = median

    return output_image


def histogram_equalization(image):
    max_pixel_value = 256
    vector_bin = np.zeros(max_pixel_value, dtype = int)

    output_image = np.zeros(image.shape, dtype = float)

    # Calculate the histogram
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            vector_bin[image[i][j]] += 1 

    # Calculate the cumulative histogram
    cumulative = 0
    for i in range(vector_bin.shape):
        cumulative += vector_bin[i]
        vector_bin[i] = cumulative 
    
    # Calculate the histogram equalisation    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output_image[i][j] = vector_bin[image[i][j]] * ((Max_pixel_value - 1) / (image.shape[0] * image.shape[1]))

    return output_image

    



# Call Main function
main()