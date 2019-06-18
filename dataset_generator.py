# Authors: Gabriel Mattheus Bezerra Alves de Carvalho and Wallace Cruz de Souza
# SCC0251 - Processamento de Imagens
# 2019, Semester 1
# Final Project: Face Detection  

import numpy as np
import pandas as pd
import math
import imageio
import matplotlib.pyplot as plt
import os
from skimage import io
import csv

from general import *

def main():
    for filename in os.listdir('images'):
        image = io.imread(filename, as_gray = True).astype(int)

        # Apply 2D Median Filter
        image = two_d_median_filter(3, image).astype(int)

        # Apply Histogram Equalization
        image = histogram_equalization(image)

        # Apply sobel algorithm
        norm_image = normalize(image, 255)
        sobel_image = sobel_operator(image)

        # Apply Edge Tracking Algorithm
        edge_tracking_algorithm(image)


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