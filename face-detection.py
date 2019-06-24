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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

#from general import two_d_median_filter, histogram_equalization, sobel_operator, edge_tracking_algorithm, normalize
from general import *

def main():
    #print("Type the name of the image...")

    # Reading filename and opening the image
    #filename = str(input()).rstrip()
    # Original image
    filename = "BioID_0001.pgm"
    #image_input = io.imread(filename)
    # Grayscale image
    image = io.imread(filename, as_gray = True).astype(int)
    image_input = io.imread(filename, as_gray = True).astype(int)
    # Ploting image
    plt.imshow(image_input)
    plt.show()
    # Apply 2D Median Filter
    median_image = two_d_median_filter(3, image).astype(int)
    # imageio.imwrite("median.jpg", median_image)

    # Apply Histogram Equalization
    histogram_image = histogram_equalization(median_image)
    # imageio.imwrite("equalization.jpg", histogram_image)

    
    #norm_image = normalize(image, 255)
    sobel_image = sobel_operator(histogram_image)
    imageio.imwrite("sobel.jpg", sobel_image)

    
    #plt.imshow(sobel_image)
    #plt.show()

    # Get the MLP Classifier
    mlp, x, a = train_classifier()
    
    # Apply Edge Tracking Algorithm
    regions = edge_tracking_algorithm(sobel_image, 1, mlp, a)

    output_image = define_face(image_input, regions[len(regions)-1][0][0], regions[len(regions)-1][2][0], regions[len(regions)-1][2][1], regions[len(regions)-1][1][1])
    # Ploting image
    plt.imshow(output_image)
    plt.show()
    # write the result image
    imageio.imwrite("result.jpg", output_image)
    return True    
                    

def train_classifier():
    dataset = pd.read_csv('dataset.data', sep = ',').values
    #print(dataset)
    #x = dataset[:, [0, 1, 2, 3]]
    #y = dataset[:, 4]
    x = dataset[:, 0:4]
    a = x
    y = dataset[:, 4]
    #print(x)
    #print(y)
    # Train the MLP Classifier
    x = normalize(x, 1)
    mlp = MLPClassifier(solver = 'adam', hidden_layer_sizes=(10,), activation = 'logistic', max_iter = 200, tol = 1e-4, learning_rate_init = 0.001)
    mlp = mlp.fit(x, y)
    return mlp, x, a

    '''
    skf = StratifiedKFold(n_splits=10)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, shuffle = True, stratify = y)
    mlp = MLPClassifier(solver = 'adam', hidden_layer_sizes=(100,), activation = 'logistic', max_iter = 400, tol = 1e-4, learning_rate_init = 0.001)
    mlp = mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    print("Acuracia = ", score)
    print(np.max(y_pred))
    print(np.mean(y_pred))

    acuracia = []
    for train_index, test_index in skf.split(x, y):
        x_train = x[train_index]
        y_train = y[train_index]
        
        x_test = x[test_index]
        y_test = y[test_index]
        
        y_pred = mlp.predict(x_test)
        print(y_pred)
        #for(i in range(len(y_pred))):
        #    if(y_pred[i] == 1 or y_pred[i])
        score = mlp.score(x_test, y_test)
  
        acuracia.append(score)
    print("Acuracia = ", np.mean(acuracia))'''
    #mlp = MLPClassifier(solver = 'adam', hidden_layer_sizes=(4,), activation = 'logistic', max_iter = 300, tol = 1e-4, learning_rate_init = 0.001)
    #mlp = mlp.fit(x, y)
    
    return mlp

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

main()