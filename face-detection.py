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
    image = io.imread(filename, as_gray = True).astype(int)

    # Apply 2D Median Filter
    image = two_d_median_filter(3, image).astype(int)

    # Apply Histogram Equalization
    image = histogram_equalization(image)
    
    norm_image = normalize(image, 255)
    sobel_image = sobel_operator(image)
    #print(sobel_image)
    plt.imshow(sobel_image)
    plt.show()

    # write the result image
    imageio.imwrite("result.jpg", norm_image)
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
    for i in range(vector_bin.shape[0]):
        cumulative += vector_bin[i]
        vector_bin[i] = cumulative 
    
    # Calculate the histogram equalisation    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output_image[i][j] = vector_bin[image[i][j]] * ((max_pixel_value - 1) / (image.shape[0] * image.shape[1]))

    return output_image

    
# Normalizing the image into a range of (0, value)
def normalize(img, value):
    img_norm = np.zeros(img.shape)
    imin = np.min(img)
    imax = np.max(img)

    img_norm = (img - imin)/(imax - imin)
    img_norm = (img_norm * value)
    return img_norm

def sobel_operator(image):
    '''
    gx_aux = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy_aux = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = np.empty(shape = image.shape, dtype = float)
    gy = np.empty(shape = image.shape, dtype = float)
    #gy = gy_aux * image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            gx[x][y] = gx_aux * image[x][y]
            gy[x][y] = gy_aux * image[x][y]
'''
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output_image = np.zeros(shape = image.shape)

    for i in range(1, image.shape[0] - 2):
        for j in range(1, image.shape[1] - 2):
            s1 = np.sum(np.sum(gx * image[i: i+3, j: j+3]))
            s2 = np.sum(np.sum(gy * image[i: i+3, j: j+3]))

            output_image[i+1, j+1] = np.sqrt(s1**2 + s2**2)
    
    return output_image

# Call Main function
main()