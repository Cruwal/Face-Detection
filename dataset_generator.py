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
        image = io.imread("images/" + filename, as_gray = True).astype(int)

        # Apply 2D Median Filter
        image = two_d_median_filter(3, image).astype(int)

        # Apply Histogram Equalization
        image = histogram_equalization(image)

        # Apply sobel algorithm
        sobel_image = sobel_operator(image)

        integral_image = integral_image_algorithm(sobel_image)

        
        mlp = 0
        mode = 0
        edge_tracking_algorithm(sobel_image, integral_image, mode, mlp)

main()