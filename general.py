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
def edge_tracking_algorithm(image, mode, b):
    print("Searching faces...")
    normalized_image = normalize(image, 1)
    #print(normalized_image)
    counter = 0
    regions = []
    for x in range(0, image.shape[0], 25):
        if(counter > 100 and mode == 1):
            break
        for y in range(0, image.shape[1], 25):
            if(counter > 100 and mode == 1):
                break
            if(image[x][y] != 0):
                top_left = (x, y)
                for j in range(y, image.shape[1]):
                    if((image[x][j] != 0) and ((j - y) > 58)):
                        top_right = (x, j)
                        for i in range(x, image.shape[0]):
                            if((image[i][j] != 0) and ((i - x) > 43)):
                                bottom_left = (i, y) #image[i][j]
                                bottom_right = (i, j) #image[x][j]
                                subwindow = image[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1]
                                normalized_subwindow = normalized_image[top_left[0]:bottom_left[0] + 1, top_left[1]:top_right[1] + 1]
                                #print(subwindow.shape)
                                if(subwindow.shape[0] > 43 and subwindow.shape[1] > 58):
                                    mean = np.mean(subwindow)
                                    if(mean != 0):
                                        features = feature_extraction(normalized_subwindow)
                                        #features2 = feature_extraction2(integral_image, top_left, top_right, bottom_left, bottom_right)
                                        print("Feature padrao: ")
                                        print(features)

                                        #print("\nFeature nova: ")
                                        #print(features2)
                                        
                                        if(mode == 0):      # Classify manually subwindow
                                            classify_manually_subwindow(subwindow, features)
                                        elif(mode == 1):    # Check if the result is a face
                                            #features = np.asarray(a[counter]).reshape(1, -1)
                                            #print(counter)
                                            #features = features.reshape(1, -1)
                                            features = features.reshape(1, 4)
                                            counter += 1
                                            #print(features.shape)
                                            #print(features)
                                            #print(b)
                                            b = np.append(b, features, axis = 0)
                                            #counter += 1
                                            #b = normalize(b, 1)
                                            #features = normalize(features, 1).reshape(1, -1)
                                            #features = np.asarray(b[b.shape[0] - 1]).reshape(1, -1)
                                            #a = np.delete(a, a.shape[0] - 1, 0)
                                            #features = features.reshape(1, -1)
                                            #print(features)
                                            #print(a)
                                            #r = mlp.predict(features)
                                            #print(y[0])
                                            #print("(", x, ",", y, ")")
                                            #if(r[0] == 1):
                                                # Ploting image
                                                #plt.imshow(subwindow)
                                                #plt.show()
                                            print("Analysing ", counter, "st subwindow")
                                            region = [top_left, top_right, bottom_left, bottom_right]
                                            regions.append(region)
                                                
    if(mode == 1):  # Predicting
        b = normalize(b, 1)
        a = b[b.shape[0] - counter: b.shape[0], :]  # Added entries
        b = b[0:b.shape[0] - counter, :]            # Dataset values
        
        return regions, a, b                       # Regions, New entries, Original entries

# Generate the integral image for feature extraction
def integral_image_algorithm(image):
    new_image = np.zeros(image.shape, dtype=int)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            new_image[x][y] = np.sum(image[0 : x + 1, 0 : y + 1])

    return new_image        

def sum_pixel(integral_image, x, y, width, height):
    sum_value = integral_image[x + width - 1, y + height - 1]    #L4

    if x > 0:   # L3
        sum_value = sum_value - integral_image[x - 1, y - 1 + height]
    if y > 0:   # L2
        sum_value = sum_value - integral_image[x - 1 + width, y - 1]
    if (x > 0 and y > 0):   # L1
        sum_value = sum_value + integral_image[x - 1, y - 1]

    return sum_value

def feature_extraction2(integral_image, top_left, top_right, bottom_left, bottom_right):
    width = bottom_left[0] - top_left[0] + 1
    height = top_right[1] - top_left[1] + 1
    x_middle = width // 2
    y_middle = height // 2

    # used in odd matrix (only in diagonal feature)
    x_rest = bottom_left[0] - width + x_middle
    y_rest = top_right[1] - height + y_middle

    all_rectangle = sum_pixel(integral_image, top_left[0], top_left[1], width, height)
    print(all_rectangle)

    #horizontal feature: sum of all rectangle - sum of bottom part
    horizontal_feature = all_rectangle - sum_pixel(integral_image, top_left[0] + x_middle, top_left[1], x_middle, height)

    # vertical feature: sum of all rectangle - sum of left part
    vertical_feature = all_rectangle - sum_pixel(integral_image, top_left[0], top_left[1], width, y_middle)

    # diagonal feature: sum all rectangle - sum of primary diagonal
    diag = sum_pixel(integral_image, top_left[0], top_left[1], x_middle, y_middle)
    diag += sum_pixel(integral_image, top_left[0] + x_middle, top_left[1] + y_middle, x_rest, y_rest)
    diag_feature = all_rectangle - diag

    #Three vertical feature: sum all rectangle - sum of side parts
    three_div = (height + 1) // 3
    y_rest = top_right[1] - height + 2 * three_div

    side = sum_pixel(integral_image, top_left[0], top_left[1], width, three_div)
    print("Parte1: ")
    print(side)
    print("Posicao parte2: ")
    print(top_left[1] + 2 * three_div)
    side += sum_pixel(integral_image, top_left[0], top_left[1] + 2 * three_div, width, y_rest)
    three_vertical_feature = all_rectangle - side

    result = np.array([horizontal_feature, vertical_feature, diag_feature, three_vertical_feature])
    print(result)
    return result

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
    return result


# Normalizing the image into a range of (0, value)
def normalize(img, value):
    img_norm = np.zeros(img.shape)
    imin = np.min(img)
    imax = np.max(img)

    img_norm = (img - imin)/(imax - imin)
    img_norm = (img_norm * value)
    return img_norm

def define_face(image, xmin, xmax, ymin, ymax):
    value = 255

    face_image = np.copy(image)

    # defining the top of rectangle
    face_image[xmin, ymin : ymax + 1] = value

    # defining the bottom of rectangle
    face_image[xmax, ymin : ymax + 1] = value

    # defining the left of rectangle
    face_image[xmin : xmax + 1, ymin] = value

    # defining the right side of rectangle
    face_image[xmin : xmax + 1, ymax] = value

    return face_image




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