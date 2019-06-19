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
    edge_tracking_algorithm(sobel_image)

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


main()