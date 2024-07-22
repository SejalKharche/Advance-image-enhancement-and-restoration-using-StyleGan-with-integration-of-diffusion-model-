import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import math
import os
import torch

def MeanFilter(image, filter_size):
#create an empty array with same size as input image output np.zeros(image.shape, np.uint8)
#creat an empty variable
  output=np.zeros(image.shape,np.unit8)
  result = 0
#deal with filter size = 3x3
if filter_size == 9:
  for j in range(1, image.shape[0]-1): 
      for i in range(1, image.shape[1]-1): 
          for y in range(-1, 2): 
              for x in range(-1, 2): 
                  result = result + image [j+y, i+x] 
                  output[j][i] = int(result / filter_size) 
                  result = 0
#deal with filter size = 5x5
elif filter_size == 25:
   for j in range(2, image.shape[0]-2): 
       for i in range(2, image.shape[1]-2): 
            for y in range(-2, 3): 
                for x in range(-2, 3): 
                    result = result + image[j+y, 1+x] 
                    output[j][i] = int(result / filter_size) 
                    result = 0
return output

def MedianFilter (image,filter_size):
#create an empty array with same size as input image

  output np.zeros(image.shape, np.uint8)
#create the kernel array of filter as some size as filter size
  filter_array [image[0][0]] filter_size

#deal with filter size 3x3
if filter size 9:
    for j in range(1, image.shape[0]-1):
        for i in range(1, image.shape[1]-1):

        filter_array[0] image [J-1, 1-1]
        
        filter_array [1] image[j, 1-1]
        
        filter_array[2] = image[1+1, 1-1]
        
        filter_array[3] = image[j-1, 1 1]
        
        filter_array [4]= image[j, 1]
        
        filter_array[5] image [j+1, 1]
        
        filter_array[6] = image[j-1, i+1]
        
        filter_array[7] = image[J, 1+1]
        
        filter_array[8] = image [j+1, i+1]

#sort the array

#put the median number into output array output[j][1] = filter_array[4]

        filter_array.sort()
        output[j][1] = filter_array[4]

#deal with filter size = 5x5

elif filter_size == 25:
    for j in range (2, image.shape[0]-2): 
        for i in range(2, image.shape[1]-2):

            filter_array[0] = image[j-2, 1-2]
            filter_array[1] image[j-1, 1-2]
            filter_array[2] = image[j, i-2]
            filter_array[3] = image[j+1, 1-2]
            filter_array[4] = image[j+2, 1-2]
            filter_array[5] = image [1-2, 1-1] 
            filter_array [6] image[j-1, 1-1]
            filter_array[7] = image[j, 1-1]
            filter_array[8] image[j+1, 1-1]
            filter_array[9] = image[j+2, 1-1]
            filter_array[10] = image[j-2, 1]
            filter_array[11] = image[j-1, i]
            filter_array[12] = 1mage[j, 1]
            filter_array[13] = image [j+1, i] 
            filter_array[14] image [j+2, 1]
            filter_array[15] = image[j-2, 1+1]
            filter_array[16] = image [1-1, 1+1]
            filter_array [17] = image[j, 1+1]
            filter_array[18] = image[j+1, 1+1] 
            filter_array[19] = image [j+2, 1+1],
            filter_array[20] = image [j-1, 1+2]
            filter_array[21] image[j-2, 1+2]
            filter_array [22] image [j, 1+2]
            filter_array[23]=image [j+1, 1+2]
            filter_array [24]=image [j+2, 1+2]
            #sort the array
            filter_array.sort()
            # put the median number into output array output[j][1] = filter_array[12]
            output[j][1] = filter_array[12]
return output
