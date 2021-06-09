# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:59:41 2021

@author: Usuario
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

mapa= cv2.imread("sample0.jpg",1)
plt.figure(figsize=(10,10))
plt.imshow(mapa[...,::-1]), plt.axis('off'), plt.title('Mapa muestra 0')


"""
b_mapa,g_mapa,r_mapa = cv2.split(mapa)
# ceros= np.zeros([])

plt.figure(figsize=(10,10))
plt.imshow(b_mapa), plt.axis('off'), plt.title('Mapa azul')
plt.figure(figsize=(10,10))
plt.imshow(g_mapa), plt.axis('off'), plt.title('Mapa verde')
plt.figure(figsize=(10,10))
plt.imshow(r_mapa), plt.axis('off'), plt.title('Mapa rojo')
"""

#%%
def simplify(img, offset):
    sliced = 0
    simple_img = np.zeros((15, 20, img.shape[2]))
    for x in range(15):
        if offset == 8 and (x == 0 or x == 19):
            sliced = 1
        else:
            sliced = 0
        for y in range(20):
            # pass the coordinates of each tile with corrections to accomodate for the grid offset
            simple_img[x, y] = classify_tile(x*16 - (offset*(1-sliced)), 
                                             y*16 - (offset*(1-sliced)), 
                                             img, sliced)
    simple_img = simple_img.astype('uint8')
    return simple_img

#%%
row_step = 2
col_step = 2
def classify_tile(x, y, img, sliced):
    px_count = 0
    total = np.zeros((1, img.shape[2]), dtype='uint32')
    freq={}
    
    corrected_height_range = int(16/(1+sliced))
    for i in range(0, corrected_height_range, row_step):
        for j in range(0, 16, col_step):
            c= img[x+i][y+j]
            c_aux= (c[0], c[1],c[2])
            if freq.get(c_aux) == None:
                freq[c_aux] = 1
            else:
                freq[c_aux]+=1
        values =  list(freq.values())
        m=0
        index=0
        
        for i in range(len(values)):
            if values[i] > m:
                m=values[i]
                index=i
                
        keys= list(freq.keys())
        result = keys[index]
        
        if img.shape[2] == 4:
            result= (result[0], result[1], result[2], 255)
        
                
    #         total += img[x+i][y+j] # [r, g, b, a]
    #         px_count += 1           
    
    # # we convert the data type before returning on the simplify() funct
    # result = total * ((1+sliced)/(px_count))
    
    return result
#%%
offset=0
# screenshot = window_capture()
img = simplify(mapa, offset)
dim = (img.shape[1] * 16, img.shape[0] * 16)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) #  interpolation = cv.INTER_AREA

plt.figure(figsize=(10,10))
plt.imshow(resized [...,::-1])

#%%
"""
We define a standar player for the agent which is the player 1 and the blue character. 
The samurai blue player is entirely blue and because of that
it can be easily simplify in the image processing. For now this part of the coding look
for the coordinates of the lifes for the blue player so the agent knows where to look when
he is killed. It also works for the rewarding in future plans. 

coordinates of the 10 lifes at the distance of 3 pixels high and 3 pixels width
from the first pixel on the bottom left side of the "leaf"

(203,8)      (203,15)
(210,8)      (210,15)
(217,8)      (217,15)
(224,8)      (224,15)
(231,8)      (231,15)

"""

    











