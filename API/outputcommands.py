# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 01:15:13 2021

@author: caleu
"""

import threading
from pynput.keyboard import Controller, Listener , Key

kb = Controller()
key_convert = {'w':'r', 'a':'m', 's':'7', 'd':'p', 'g':'z', 'h':'q', 'j':'l'}

delay = 0

def on_press(key):
    #do stuff
    x = threading.Thread(target=t_press, args=(key,))
    x.start()
    
def on_release(key):
    #do other stuff
    if key != Key.esc:
        x = threading.Thread(target=t_release, args=(key,))
        x.start()
    else:
        #stop listener
        return False
        
def t_press(key):
    # time.sleep(delay)
    k = str(key) #converting the "key" object to a string
    k = k.replace("'", "") #cleaning the "k" string
    if k in key_convert.keys():
        kb.press(key_convert[k])
        # print("converting to: ", key_convert[k])

def t_release(key):
    # time.sleep(delay)
    k = str(key)
    k = k.replace("'", "")
    if k in key_convert.keys():
        kb.release(key_convert[k])
    # print("releasing ", key)

if __name__ == "__main__":   
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
    