# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:59:41 2021

@author: Usuario
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL

mapa=cv2.imread('sample0.jpg',1)
b_mapa,g_mapa,r_mapa = cv2.split(mapa)

plt.figure(figsize=(10,10))
plt.imshow(mapa[...,::-1]), plt.axis('off')
#%%
plt.figure(figsize=(10,10))
plt.imshow(b_mapa[...,::]), plt.axis('off'), plt.title('Mapa azul')
#%%
plt.figure(figsize=(10,10))
plt.imshow(g_mapa), plt.axis('off'), plt.title('Mapa verde')
plt.figure(figsize=(10,10))
plt.imshow(r_mapa), plt.axis('off'), plt.title('Mapa rojo')

