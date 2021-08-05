import cv2 as cv
import numpy as np
from time import sleep

from .inputpreprocess import observation, Map, lives, enemy_lives, enemy_lives_2, renderObs
from .outputcommands import press, release, tap

# Use the 2 below instead of the 2 above when testing the module by itself
# from inputpreprocess import observation, Map, lives, enemy_lives
# from outputcommands import press, release, tap


def setup() -> Map:
    ''' 
    Crea el mapa con la mascara y el offst a utilizar y lo devuelve en la estructura Map 

    ## Note: As for now, offset is hardcoded as 0

    Returns: Map
    '''
    ##
    # Esta funcion debe hacer, en orden de importancia
    # - seleccionar mapa
    # - seleccionar personaje
    # - configurar controles
    ##

    # selecting map and difficulty
    tap('swing')
    for i in range(5):
        sleep(0.05)
        tap('down')
    #sleep(0.1)
    tap('swing')
    sleep(1.2)
    empty = Map(0, )
    screenshot = observation(empty, raw=True)
    map = Map(0, screenshot)
    return map

def pause() -> None:
    tap('menu')

def restart() -> None:
    '''
    Abre el menu de pausa para reiniciar la partida en caso de que el jugador o el enemigo se le acaben las vidass
    '''
    releaseAllKeys()

    WAIT = 0.05
    tap('menu')
    sleep(WAIT)
    tap('down')
    sleep(WAIT)
    tap('swing')
    # sleep(0.2)
    #print("sleep time's over")
 

def reset(map: Map):
    '''
    This funcion is called when an episode has concluded.
    It resets the environment and returns an initial observation (simplified screenshot), along with answering if 
    the player is on screen, ready to play
    
    Parameters:
        Map
    Returns:
        ndarray: 15x20 (simplified screenshot)
        start: bool. Is a player on the screen?
    '''
    initial_state, raw_ , start = observation(map)
    starting_p_lives = lives(raw_)
    starting_e_lives = enemy_lives_2(raw_)

    map.p_lives = starting_p_lives
    map.e_lives = starting_e_lives

    return initial_state, start


DELAY = 0.1

ACTIONS_DICT={0:'left', 1:'left', 2:'right', 3:'right', 4:'up', 5:'up', 6: 'down', 7:'down', 8:'jump', 9:'jump', 10:'swing'}

def step(action, map: Map):
    done = False
    restart = False  # Flag that signals whenever either the player's or the enemy's lives have reached 0, hence requiring to restart
    
    # assign to every number of action a key for execute the code. I think this can be automatized.

    # split between 0 n' 1 if action is odd or even, even is release, odd is press
    if action == 10:
        tap(ACTIONS_DICT[action])
    elif action % 2 == 1:
        press(ACTIONS_DICT[action])
    elif action % 2 == 0:
        release(ACTIONS_DICT[action])
    
    sleep(DELAY)
    new_state, raw_, _ = observation(map)

    current_p_lives = lives(raw_)
    current_e_lives = enemy_lives_2(raw_)

    #print(f'{current_p_lives = }, {current_e_lives = }')
    # Checking if any player has lost any lives
    p_lives_diff = current_p_lives - map.p_lives
    e_lives_diff = current_e_lives - map.e_lives

    # Defining rewards
    reward = 0.0
    if e_lives_diff == -1:
        reward += 5
        done = True
    if p_lives_diff == -1:
        reward -= 1
        done = True
    if e_lives_diff - p_lives_diff == 0:
        reward -= 0.02
    if (current_p_lives < 3) or (current_e_lives < 3):
        restart = True

    map.p_lives = current_p_lives
    map.e_lives = current_e_lives
    return new_state, reward, done, restart

def render(obs):
    renderObs(obs)

def releaseAllKeys():
    sleep(0.05)
    for action in range(5):
        release(ACTIONS_DICT[action*2])
        sleep(0.002)

if __name__ == '__main__':
    sleep(2)
    restart()

