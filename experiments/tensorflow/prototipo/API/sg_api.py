import cv2 as cv
import numpy as np
from time import time
from inputpreprocess import observation, Map
from outputcommands import press, release

def setup():
    '''
    Toma el control del juego, navega el menu, escoje un mapa y crea la mascara de este.

    Returns:
    Map (contiene la mascara del mapa seleccionado y su offset)
    '''
    #secuencia de inputs para navegar el juego
    # - configurar controles
    # - seleccionar personaje
    # - seleccionar mapa
    
    #screenshot = window_capture('Samurai Gunn')
    #map = Map(0, screenshot)
    return map

def reset(map: Map):
    '''
    This funcion is called when an episode has concluded.
    It resets the environment and returns an initial observation (simplified screenshot).
    
    Parameters:
        Map
    Returns:
        ndarray (simplified screenshot)
    '''
    #

    initial_state = observation(map)

   
    return initial_state

def step(action):    
    return new_state, reward, done

'''
observation = env.reset() 
# reinicia el ambiente y retorna un screenshot

observation_, reward, done, info = env.step(action) 
# toma como parametro una accion del espacio de acciones, 
# retorna:
#  un screenshot del nuevo estado DESPUES de haber tomado la accion
#  la recompenza por haber alcansado este nuevo estado
#  un booleano indicando si se alcansó el fin del episodio
#  información util para debug
 
 
'''
