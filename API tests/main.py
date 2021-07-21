import cv2 as cv
import numpy as np
from time import time
from windowcapture import window_capture
from inputpreprocess import simplify_2, map_mask, lives, enemy_lives


if __name__ == "__main__":      
    frame_count = 1
    cumulative_fps = 0
    ss_counter = 0  # "screenshot counter"
    
    # offset will depend on the map
    # for Ice cube, offset is 0
    offset = 0
    mk = None
    once = True
    
    while(True):
        prev_time = time()
        
        #screenshot = cv.imread('sample0.jpg')
        screenshot = window_capture('Samurai Gunn')
        
        img = simplify_2(screenshot, offset=offset, mask=mk)
        
        # re-calculate mask every so often
        if once: 
            mk = map_mask(simplify_2(screenshot, offset=offset))
            once = False
        
        dim = (img.shape[1] * 16, img.shape[0] * 16)
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA) #  interpolation = cv.INTER_AREA
    
        cv.imshow('Simplified', resized)
        #cv.imshow('Screenshot', screenshot)
        #print('FPS {}'.format(1 / (time() - prev_time)))
        
        delta_time = time() - prev_time
        if delta_time == 0:
            delta_time = 1
            
        cumulative_fps += 1 / delta_time
        if frame_count % 128 == 0:
            print('Avg FPS {}'.format(cumulative_fps/frame_count))
        frame_count += 1
        
        pj_lives = lives(screenshot)
        if frame_count % 128 == 0:
            print("Player lives:",  pj_lives)
            
            enemy = enemy_lives(screenshot)
            cv.imshow('Area of enemy lives', enemy)
            frame_count = 1
            cumulative_fps = 0
        
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
    
