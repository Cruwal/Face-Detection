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
    print("Type the name of the image...")

    # Reading filename and opening the image
    filename = str(input()).rstrip()
    
    # Original image
    # filename = "BioID_0000.pgm"
    
    # Grayscale image
    image = io.imread(filename, as_gray = True).astype(int)
    image_input = io.imread(filename, as_gray = True).astype(int)
    
    # Ploting image
    plt.imshow(image_input, cmap="gray")
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

    integral_image = integral_image_algorithm(sobel_image)
    
    #plt.imshow(sobel_image)
    #plt.show()

    # Read dataset
    x, y = read_dataset()
    # Apply Edge Tracking Algorithm
    regions, a, new_x = edge_tracking_algorithm(sobel_image, integral_image, mode = 1, b = x)
    
    mlp = MLPClassifier(solver = 'adam', hidden_layer_sizes=(30,10), activation = 'logistic', max_iter = 200, tol = 1e-4, learning_rate_init = 0.001)
    mlp = mlp.fit(new_x, y)

    print(a)
    y_pred = mlp.predict(a)

    min_area = float("inf")
    regions_to_show = []
    for i in range(len(y_pred)):
        if(y_pred[i] == 1):
            regions_to_show.append(regions[i])
            area = (regions_to_show[len(regions_to_show)-1][2][0] - regions_to_show[len(regions_to_show)-1][0][0]) * (regions_to_show[len(regions_to_show)-1][1][1] - regions_to_show[len(regions_to_show)-1][2][1])
            if area < min_area:
                min_area = area
                xmin = regions_to_show[len(regions_to_show)-1][0][0]
                xmax = regions_to_show[len(regions_to_show)-1][2][0]
                ymin = regions_to_show[len(regions_to_show)-1][2][1]
                ymax = regions_to_show[len(regions_to_show)-1][1][1]


    print("\n\nLISTA TAM:")
    print(len(regions_to_show)-1)
    print("\n\n")

    # Ploting the last found value
    if len(regions_to_show) == 0:
        print("\n\nCouldn't find any face!\n\n")
    else:
        output_image = define_face(image_input, xmin, xmax, ymin, ymax)
        # Ploting image
        plt.imshow(output_image, cmap="gray")
        plt.show()
        # write the result image
        imageio.imwrite("result.jpg", output_image)
    
    return True    
                    

def read_dataset():
    dataset = pd.read_csv('dataset.data', sep = ',').values
    x = dataset[:, 0:4]
    y = dataset[:, 4]
    return x, y


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



'''
Se criando dataset:
    Pega as features e insere, não precisa de MLP
Se lendo:
    Pega todas as features
    Adiciona ao dataset
    Normaliza o dataset
    Prevê os últimos 50?
    Mostra um dos últimos 50
'''

