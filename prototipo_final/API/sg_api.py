import cv2 as cv
import numpy as np
import time

# from .inputpreprocess import observation, Map, lives, enemy_lives
# from .outputcommands import press, release, tap

# Use the 2 below instead of the 2 above when testing the module by itself
from inputpreprocess import observation, Map, lives, enemy_lives
from outputcommands import press, release, tap

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
    WAIT = 0.0
    tap('menu')
    time.sleep(WAIT)
    tap('down')
    time.sleep(WAIT)
    tap('swing')
    time.sleep(1)
    print("sleep time's over")


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


DELAY = 0.07

def step(action, map: Map):
    raw = observation(map, raw=True)
    previous_p_lives = lives(raw)
    previous_e_lives = enemy_lives(raw)
    done = False
    restart = False  # Flag that signals whenever either the player's or the enemy's lives have reached 0, hence requiring to restart
    
    # assign to every number of action a key for execute the code. I think this can be automatized.
    dict_actions={0:'left', 1:'left', 2:'right', 3:'right', 4:'up', 5:'up', 6: 'down', 7:'down', 8:'jump', 9:'jump', 10:'swing'}

    # split between 0 n' 1 if action is odd or even, even is release, odd is press
    if action == 10:
        tap (dict_actions[action])
    elif action % 2 == 1:
        press(dict_actions[action])
    elif action % 2 == 0:
        release(dict_actions[action])
    
    time.sleep(DELAY)
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

if __name__ == '__main__':
    time.sleep(2)
    restart()
    # for i in range(5):
    #     tap('swing')
    #     time.sleep(0.3)

    # press('right')
    # time.sleep(1)
    # release('right')

    # press('swing')
    # time.sleep(0.1)
    # release('swing')

    # press('left')
    # time.sleep(1)
    # release('left')

    # press('right')
    # time.sleep(0.1)
    # release('right')
