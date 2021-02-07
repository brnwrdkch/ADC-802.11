#!/usr/bin/env python
# coding: utf-8

# In[50]:


import cv2
import numpy as np
total_bit=[]
img = cv2.imread('C:\\Users\\adc\\Desktop\\adc2\\opencv\\smile.png')
_, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
binery=[[[1 if pixel==255 else pixel for pixel in row]for row in matrix]for matrix in bw_img]

for i in binery:
    for j in i:
        for k in j:
            total_bit.append(k)

        

#for seperate sample from total_bit


#lower_band=0
#upper_band=8

#matix_sample=[]
#while upper_band<len(total_bit):
    #sampel=[]
    #for i in total_bit[lower_band:upper_band]:
       #sampel.append(i)
    #matix_sample.extend([sampel])
    #lower_band+=8
    #upper_band+=8

