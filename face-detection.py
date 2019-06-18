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
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

#from general import two_d_median_filter, histogram_equalization, sobel_operator, edge_tracking_algorithm, normalize
from general import *

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
    
    #plt.imshow(sobel_image)
    #plt.show()

    # Apply Edge Tracking Algorithm
    edge_tracking_algorithm(image)

    # Get the MLP Classifier
    clf = train_classifier()

    # write the result image
    imageio.imwrite("result.jpg", sobel_image)
    return True    
                    

def train_classifier():
    dataset = pd.read_csv('dataset.data', sep = ',')
    x = dataset[:, [0, 1, 2, 3]]
    y = dataset[:, 4]
    # Train the MLP Classifier
    mlp = MLPClassifier(solver = 'sgd', hidden_layer_sizes=(4,), activation = 'logistic', max_iter = 200, tol = 1e-4, learning_rate_init = 0.001)
    mlp = mlp.fit(x, y)
    
    return mlp

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
    return result


main()