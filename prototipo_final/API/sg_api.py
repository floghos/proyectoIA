import cv2 as cv
import numpy as np
from time import time

from .inputpreprocess import observation, Map, lives, enemy_lives
from .outputcommands import press, release

# Use the 2 below instead of the 2 above when testing the module by itself
# from inputpreprocess import observation, Map
# from outputcommands import press, release

def restart():
    '''
    Toma el control del juego, navega el menu, escoje un mapa y crea la mascara de este.

    Returns:
    Map (contiene la mascara del mapa seleccionado y su offset)
    '''
    #secuencia de inputs para navegar el juego (opcional)
    # - configurar controles
    # - seleccionar personaje
    # - seleccionar mapa

    ##
    # Como minimo debemos:
    #   -presionar start
    #   -seleccionar restart
    #   -esperar un momento (~1 seg) antes de crear map (para evitar que la animacion del menu afecten la mascara)
    ##

    empty = Map(0, )
    screenshot = observation(empty, raw=True)
    map = Map(0, screenshot)
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
    initial_state, _ = observation(map)
 
    return initial_state

def step(action, map: Map):
    raw = observation(map, raw=True)
    previous_p_lives = lives(raw)
    previous_e_lives = enemy_lives(raw)
    done = False
    restart = False  # Flag that signals whenever either the player's or the enemy's lives have reached 0, hence requiring to restart
    
    ##
    # perform action
    ##

    new_state, raw_ = observation(map)

    current_p_lives = lives(raw_)
    current_e_lives = enemy_lives(raw_)
    p_lives_diff = current_p_lives - previous_p_lives
    e_lives_diff = current_e_lives - previous_e_lives

    reward = 0.0

    if e_lives_diff == -1:
        reward += 1
        done = True

    if p_lives_diff == -1:
        reward -= 1
        done = True

    if e_lives_diff - p_lives_diff == 0:
        reward -= 0.01

    if (current_p_lives == 0) or (current_e_lives == 0):
        restart = True

    return new_state, reward, done, restart

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
