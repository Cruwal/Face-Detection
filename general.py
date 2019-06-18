# Authors: Gabriel Mattheus Bezerra Alves de Carvalho and Wallace Cruz de Souza
# SCC0251 - Processamento de Imagens
# 2019, Semester 1
# Final Project: Face Detection  

import numpy as np
import pandas as pd
import math
import imageio
import matplotlib.pyplot as plt
from skimage import io
import csv

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


# Apply the sobel operator at a given image
def sobel_operator(image):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output_image = np.zeros(shape = image.shape)

    for i in range(1, image.shape[0] - 2):
        for j in range(1, image.shape[1] - 2):
            s1 = np.sum(np.sum(gx * image[i: i+3, j: j+3]))
            s2 = np.sum(np.sum(gy * image[i: i+3, j: j+3]))

            output_image[i+1, j+1] = np.sqrt(s1**2 + s2**2)
    
    threshold = 100 #%varies for application [0 255]
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            #output_image[x][y] = np.max(output_image[x][y], threshold)
            if(output_image[x][y] < threshold):
                output_image[x][y] = 0
	
    return output_image


# Apply Edge Tracking Algorithm
def edge_tracking_algorithm(image, mode):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if(image[x][y] != 0):
                top_left = (x, y)
                for j in range(y, image.shape[1]):
                    if(image[x][j] != 0):
                        if((j - y) > 50):
                            top_right = (x, j)
                            for i in range(x, image.shape[0]):
                                if(image[i][j] != 0):
                                    if((i - x) > 50):
                                        bottom_left = (i, y) #image[i][j]
                                        bottom_right = (i, j) #image[x][j]
                                        subwindow = image[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1]
                                        print(subwindow.shape)
                                        if(subwindow.shape[0] > 50 and subwindow.shape[1] > 50):
                                            features = feature_extraction(subwindow)
                                            mean = np.mean(subwindow)
                                            if(mean != 0):
                                                if(mode == 0):      # Classify manually subwindow
                                                    classify_manually_subwindow(subwindow, features)

# Generate the integral image for feature extraction
def integral_image_algorithm(image):
    new_image = np.zeros(image.shape, dtype=int)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            new_image[x][y] = np.sum(image[0 : x + 1, 0 : y + 1])

    return new_image        


def feature_extraction(image):
    integral_image = integral_image_algorithm(image)

    horizontal_middle = image.shape[0] // 2
    vertical_middle = image.shape[1] // 2

    # horizontal feature
    aux = np.sum(integral_image[0 : horizontal_middle, 0 : image.shape[1]]) 
    aux1 = np.sum(integral_image[horizontal_middle : image.shape[0], 0 : image.shape[1]])
    horizontal_feature = aux - aux1

    # vertical feature
    aux = np.sum(integral_image[0 : image.shape[0], 0 : vertical_middle])
    aux1 = np.sum(integral_image[0 : image.shape[0], vertical_middle : image.shape[1]])
    vertical_feature = aux1 - aux

    # Sum first diagonal
    diag = np.sum(integral_image[0 : horizontal_middle, 0 : vertical_middle])
    diag1 = np.sum(integral_image[horizontal_middle : image.shape[0], vertical_middle : image.shape[1]])
    diag = diag + diag1
    
    # Sum second diagonal
    diag2 = np.sum(integral_image[0 : horizontal_middle, vertical_middle : image.shape[1]])
    diag1 = np.sum(integral_image[horizontal_middle : image.shape[0], 0 : vertical_middle])
    diag1 = diag1 + diag2

    # Subtract diagonals
    diag_feature = diag1 - diag

    # Tree vertical
    tree_div = image.shape[1] // 3
    vertical_p1 = np.sum(integral_image[0 : image.shape[0], 0 : tree_div])
    vertical_center = np.sum(integral_image[0 : image.shape[0], tree_div : 2 * tree_div + 1])
    vertical_p2 = np.sum(integral_image[0 : image.shape[0], 2 * tree_div + 1 : image.shape[1]])

    tree_vertical_feature = vertical_center - vertical_p1 + vertical_p2

    result = np.array([horizontal_feature, vertical_feature, diag_feature, tree_vertical_feature])
    print(result)
    return result


# Normalizing the image into a range of (0, value)
def normalize(img, value):
    img_norm = np.zeros(img.shape)
    imin = np.min(img)
    imax = np.max(img)

    img_norm = (img - imin)/(imax - imin)
    img_norm = (img_norm * value)
    return img_norm






def classify_manually_subwindow(subwindow, features):
    fp = open(r"dataset.data", "a")
    
    # Ploting image
    plt.imshow(subwindow)
    plt.show()

    # Generating line on dataset
    for i in range(len(features)):
        fp.write(str(features[i]))
        fp.write(",")

    # Assigning the subregion as a face or not
    print("Is it a face? y/n")
    c = str(input())
    if(c == "y"):
        fp.write("1 \n")
    else:
        fp.write("0 \n")

    fp.close()
    return True