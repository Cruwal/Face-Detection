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

        test_image = np.array([[1, 3, 7, 5, 1], [12, 4, 8, 2, 1], [0, 14, 16, 9, 1], [5, 11, 6, 10, 1]])

        integral_image = integral_image_algorithm(test_image)
        print(integral_image)
        # sum_pixel(integral_image, 0, 0, 4, 4)
        TL = (1, 1)
        TR = (1, 4)
        BL = (3, 1)
        BR = (3, 4)

        feature_extraction2(integral_image, TL, TR, BL, BR)

        # Ploting image
        # plt.imshow(sobel_image)
        # plt.show()

        mlp = 0
        mode = 0
        # defined_image = define_face(sobel_image, 50, 200, 50, 300)
        # imageio.imwrite("result.jpg", defined_image)
        # edge_tracking_algorithm(sobel_image, integral_image, mode, mlp)


main()