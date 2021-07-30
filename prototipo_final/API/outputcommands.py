#import threading
from pynput.keyboard import Controller, Key
from time import sleep
kb = Controller()
key_dict = {'left':'a', 'right':'d', 'up':'w', 'down':'s', 'jump':'j', 'swing':'k', 'shoot':'l', 'menu': Key.esc, 'back':'o'}

delay = 0

# Threaded version might not be needed for the API
# def press(key):
#     #do stuff
#     x = threading.Thread(target=t_press, args=(key,))
#     x.start()
    
# def release(key):
#     #do other stuff
#     x = threading.Thread(target=t_release, args=(key,))
#     x.start()
        
def press(key):
    # time.sleep(delay)
    if key in key_dict.keys():
        kb.press(key_dict[key])
        # print("converting to: ", key_dict[k])
    else:
        try:
            kb.press(key)
        except:
            print('No se pudo presionar "{}"'.format(key))


def release(key):
    # time.sleep(delay)
    if key in key_dict.keys():
        kb.release(key_dict[key])
        # print("releasing ", key)
    else:
        try:
            kb.release(key)
        except:
            print('No se pudo soltar "{}"'.format(key))

def tap(key):
    # time.sleep(delay)
    if key in key_dict.keys():
        #print ('tapping', key)
        kb.press(key_dict[key])
        sleep(0.05)
        kb.release(key_dict[key])
    else:
        try:
            kb.tap(key)
        except:
            print('No se pudo hacer tap "{}"'.format(key))
