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
        #norm_image = normalize(image, 255)
        sobel_image = sobel_operator(image)
        # write the result image
        #imageio.imwrite("result.jpg", sobel_image)
        #print(sobel_image)
        # Apply Edge Tracking Algorithm

        # Ploting image
        plt.imshow(sobel_image)
        plt.show()

        edge_tracking_algorithm(sobel_image, mode = 0)


main()