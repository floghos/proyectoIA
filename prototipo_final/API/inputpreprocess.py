import cv2 as cv
import numpy as np
from .windowcapture import window_capture  # use this when running from main.py
# from windowcapture import window_capture  #use when testing this module by itself or from sg_api.py

class Map:
    def __init__(self, offset, screenshot=None) -> None:
        self.offset = offset
        self.e_lives = 0
        self.p_lives = 0
        if screenshot is None:
            self.mask = None
        else:
            self.mask = map_mask(simplify_2(screenshot))
    
def lives(img):
    '''
    We settled a standar player for the agent which is the player 1 and the green character. 
    His lives are static in the top left area of the map. For now this part of the coding seek
    a pair of coordinates on the image matrix that will represent each of the ten lives for the green player, 
    so the agent know the lives he has at every moment. It also works for the rewarding in future plans. 

    Coordinates of the 10 lives at the distance of 3 pixels high and 3 pixels width
    from the first pixel on the top left side of the "leaves" are:

    (7,8)       (7,15)
    (14,8)      (14,15)
    (21,8)      (21,15)
    (28,8)      (28,15)
    (35,8)      (35,15)

    For green characters leafs are on 208 in green intensity
    '''
    img_without_alpha= img[...,:3]
    b_mapa,g_mapa,r_mapa = cv.split(img_without_alpha)

    life = 0      #contador de vidas

    for fila in range (7,36,7):
        for col in range (8,16,7):
            intensity = g_mapa[fila,col]

            if intensity == 208:
                life += 1
    return(life)


TEMPLATE = cv.imread("prototipo_final/API/char_templates/template_mini.bmp",0)

def enemy_lives(img):
    '''
    For finding the enemy lives, we used the cv.matchTemplate function which review the template in the corresponding image
    but in gray scale, which was previously converted. The template was created manually with black background and 
    the white skull, that represents the lives of the enemy in survival mode. We reduced the area were the template
    was evaluated from the entire image to the section were the lives of the enemy stayed static. For avoiding 
    multiple detection the threshold for the mathcTemplate was 0.6 and determined arbitray according to various scenarios.     
    '''
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    count_enemy = 0

    line=np.zeros((30,100),dtype='uint8')
    for fila in range(0,30):
        for col in range(0,100):
            line[fila,col]=gray_img[fila+210, col+110]    
    locations = cv.matchTemplate(line, TEMPLATE, cv.TM_CCOEFF_NORMED)
    thresh = 0.3
    locations_w = np.where(locations >= thresh)
    locations_f = list(zip(*locations_w[::-1]))    
    count_enemy = len(locations_f)    
       
    #print('Enemy lives:', count_enemy)    
    return count_enemy


def enemy_lives_2(img) -> int:
    img2 = img.copy()
    # Yes, these 2 inner functions are almost identical... and yes, they could be made into a single one
    # but I think this way it's more understandable, so for now I'll just leave it like this

    def checkLeft(t: int) -> bool:
        t = 5 - t
        origin1 = (217, 123 + 8*t)
        px_set = ((0, 4), (6, 0), (14, 4))
        for p in px_set:
            coords = (origin1[0] + p[0], origin1[1] + p[1])
            pixel = img[coords]
            pixel = tuple(pixel[:3])
            if pixel != (255, 255, 255):
                return False
        return True

    def checkRight(t: int) -> bool:
        t = 5 - t
        origin2 = (217, 196 - 8*t)
        px_set = ((0, -4), (6, 0), (14, -4))
        for p in px_set:
            coords = (origin2[0] + p[0], origin2[1] + p[1])
            pixel = img[coords]
            pixel = tuple(pixel[:3])
            if pixel != (255, 255, 255):
                return False
        return True

    for i in range(5):
        l = 5 - i
        #this ask the question: does the enemy have l lives?
        if checkLeft(l) and checkRight(l):
            return l

    return 0

def map_mask(img) -> np.ndarray:
    '''Makes a matrix with boolean values, marking with 1 all tiles that are
    a solid block on the map, and a 0 for empty air tiles.
    
    Parameters:
        img: simplified img, a matrix of 15 x 20
    '''
    mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    size = mask.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if mask[i][j] < 70 or mask[i][j] > 230:
                mask[i][j] = 1
            elif mask[i][j] > 100 and mask[i][j] < 190:
                mask[i][j] = 0
    return mask

def observation(map: Map, raw=False):
    '''Returns a simplified view of the screen in a 20x15 nparray.
    If raw is given as True, the raw screenshot of the game is returned instead.
    '''
    screenshot = window_capture('Samurai Gunn')

    if raw:
        return screenshot

    # "player key color" is a distinctive color for the rat-like player
    PKC = (65, 191, 97)

    # "enemy key color" is a distinctive color for the enemy ninja bot in survival mode
    EKC = (11, 11, 13)

    # "mixed key color" is the average color of both, used to indicate both are on the same tile
    aux = np.zeros(3, dtype='uint8')
    for i in range(3):
        aux[i] = (PKC[i] + EKC[i]) / 2
    MKC = tuple(aux)

    
    obs = np.zeros((15, 20), dtype=np.float16)
    const = 10 # used for getting the classifying numbers to a range between 0 and 1 
    s_img = simplify_2(screenshot, offset=map.offset, mask=map.mask)
    player_on_screen = False
    for x in range(15):
        for y in range(20):
            if tuple(s_img[x, y]) == PKC:
                obs[x, y] = 5/const
                player_on_screen = True
            if tuple(s_img[x, y]) == EKC:
                obs[x, y] = 10/const
            if tuple(s_img[x, y]) == MKC:
                obs[x, y] = 7/const
            if tuple(s_img[x, y]) == (0, 0, 0):
                obs[x, y] = 1/const
            
    return obs, screenshot, player_on_screen

def simplify_2(img, offset=0, mask=None):
    '''Reduces the input image resolution by classifying the tiles on the screen and 
    reducing them to 1 px per tile.
    The tiles in every map can be aligned with a 20 x 15 grid of (16 x 16)px cells.
    Iterates over each tile, calling classify_tile() for each of them
    
    Parameters:
    img: input image
    offset: in case the map tiles do not align perfectly with the grid, an offset of 8px down does 
            the trick. (Maps with different offsets have yet to be found, 
            this should allow for different offsets in the X direction)
            Note: the offset functionality has not been properly tested yet

    mask: 20 x 15 matrix of boolean values, where a 1 means it's a solid block 
          and should won't ever contain a player, allowing us to skip checking 
          the corresponding tile.
          
    Returns:
    numpy 3D array: (15 x 20 x num_channels) numpy array
    '''
    sliced = 0
    simple_img = np.zeros((15, 20, 3))    
    for x in range(15):
        if offset == 8 and (x == 0 or x == 19):
            sliced = 1
        else:
            sliced = 0
        for y in range(20):
            # pass the coordinates of each tile with corrections to accomodate for the grid offset
            if mask is None:
                simple_img[x, y] = classify_tile_2(x*16 - (offset*(1-sliced)), 
                                                   y*16 - (offset*(1-sliced)), 
                                                   img, sliced)
                continue
            if mask[x][y] == 0:
                simple_img[x, y] = find_players(x*16 - (offset*(1-sliced)),
                                                y*16 - (offset*(1-sliced)),
                                                img, sliced)
    simple_img = simple_img.astype('uint8')
    return simple_img

def classify_tile_2(x, y, img, sliced):
    ''' Classifies the tile as defined by it's x,y coordinates according to the
    most frecuent pixel present on it.
    Used to discern between platform blocks and air blocks, and later create a
    mask with this info.
    
    Note: Tiles are (16 x 16)px
    
    Parameters:
    x, y  : coords of the upper left corner of the tile
    img: input image
    sliced: int, either 0 or 1, letting us know if it's a full tile or a half tile at the 
             upper/lower edges of the map. 
             0 means it's a full tile
             1 means it's a half tile
    Returns:
    int: Returns a single int that will classify the tile
    '''

    ROW_STEP = 2
    COL_STEP = 2
    freq = {}
    corrected_height_range = int(16/(1+sliced))
    for i in range(0, corrected_height_range, ROW_STEP):
        for j in range(0, 16, COL_STEP):
            c = img[x+i][y+j]  
            c_aux = (c[0], c[1], c[2]) # change the color array into a hashable tuple
            if freq.get(c_aux) == None:
                freq[c_aux] = 1
            else:
                freq[c_aux] += 1
    values = list(freq.values())
    m = 0
    index = 0
    for i in range(len(values)):
        if values[i] > m:
            m = values[i]
            index = i
    keys = list(freq.keys())
    result = keys[index]
    return result


PLAYER_TEMP_R = cv.imread(
    'prototipo_final/API/char_templates/splinter_needle_r_eye.png')
PLAYER_TEMP_L = cv.imread(
    'prototipo_final/API/char_templates/splinter_needle_l_eye.png')
ENEMY_TEMP_GREEN = cv.imread(
    'prototipo_final/API/char_templates/ninja_needle_green.jpg')
ENEMY_TEMP_CYAN = cv.imread(
    'prototipo_final/API/char_templates/ninja_cyan.png')

# This is just here for reference. Used in cv.matchTemplate()
# METHODS = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

def find_players(x, y, img, sliced):
    '''Used to find the characters within a tile.
    Search is conducted using openCV matchTemplate() function.
    
    Parameters:
        x, y: Coordinates of the tile to be examined.
        img: Full screenshot of the game's state
        sliced: int between [0, 1] with 0 meaning it's a full tile, 0 otherwise
        
    Return: returns a single int that corresponds with the findings

    '''
    # "player key color" is a distinctive color for the rat-like player
    #  colors in BGR format (???)
    PKC = (65, 191, 97)

    # "enemy key color" is a distinctive color for the enemy ninja bot in survival mode
    # EKC2 is used for the ninja with cyan eyes, whose color pallete is slightly different
    EKC = (11, 11, 13)
    EKC2 = (20, 20, 32)
    

    # "mixed key color" is the average color of both, used to indicate both are on the same tile
    avg = np.zeros(3, dtype='uint8')
    for i in range(3):
        avg[i] = (PKC[i] + EKC[i]) / 2
    MKC = avg

    
    ROW_STEP = 2
    COL_STEP = 2   
    player_flag = False
    enemy_flag = False
    corrected_height_range = int(16/(1+sliced))

    # A preliminary search is done to check if there is a chance of a character being present
    # using the character's "key colors"
    for i in range(0, corrected_height_range, ROW_STEP):
        for j in range(0, 16, COL_STEP):
            aux = img[x+i][y+j][:3]
            if tuple(aux) == PKC:
                player_flag = True
                #print('player flag set')
            if tuple(aux) == EKC:
                enemy_flag = True
                #print('enemy flag set')
            if tuple(aux) == EKC2:
                enemy_flag = True
                #print('enemy flag set')
                
    tile = np.zeros((16, 16, 3), dtype='uint8')
    if player_flag or enemy_flag:
        for i in range(0, corrected_height_range):
            for j in range(0, 16):
                tile[i][j] = img[x+i][y+j][:3]
            
                
    P_THRESHOLD = 0.6
    E_THRESHOLD = 0.3

    min_val_p, max_val_p, min_loc_p, max_loc_p = (0, 0, (0,0), (0,0))
    min_val_e, max_val_e, min_loc_e, max_loc_e = (0, 0, (0,0), (0,0))
    
    if player_flag:
        search_p1 = cv.matchTemplate(tile, PLAYER_TEMP_R, cv.TM_CCOEFF_NORMED)
        min_val_p1, max_val_p1, min_loc_p, max_loc_p = cv.minMaxLoc(search_p1)
        
        search_p2 = cv.matchTemplate(tile, PLAYER_TEMP_L, cv.TM_CCOEFF_NORMED)
        min_val_p2, max_val_p2, min_loc_p, max_loc_p = cv.minMaxLoc(search_p2)
        
        # used for CCOEFF and CCORR methods where the highest value represents 
        # the best match
        max_val_p = max_val_p1 if (max_val_p1 > max_val_p2) else max_val_p2
        
        # used for SQDIFF methods, where the lowest value represents the best
        # match
        # min_val_p = min_val_p1 if (min_val_p1 < min_val_p2) else min_val_p2
        
    if enemy_flag:
        search_e = cv.matchTemplate(tile, ENEMY_TEMP_GREEN, cv.TM_CCOEFF_NORMED)
        min_val_e1, max_val_e1, min_loc_e1, max_loc_e1 = cv.minMaxLoc(search_e)

        search_e2 = cv.matchTemplate(tile, ENEMY_TEMP_CYAN, cv.TM_CCOEFF_NORMED)
        min_val_e2, max_val_e2, min_loc_e2, max_loc_e2 = cv.minMaxLoc(search_e2)

        max_val_e = max_val_e1 if (max_val_e1 > max_val_e2) else max_val_e2

        
    if max_val_p > P_THRESHOLD and max_val_e < E_THRESHOLD:
        # Player found in tile
        return PKC
    elif max_val_p < P_THRESHOLD and max_val_e > E_THRESHOLD:
        # Enemy found in tile
        #print(max_val_e)
        #cv.imwrite('tile.png', tile)
        return EKC
    elif max_val_p > P_THRESHOLD and max_val_e > E_THRESHOLD:
        # Both player and enemy found in the same tile
        return MKC
    else:
        return (255, 255, 255)

def renderObs(observation):
    img = np.zeros((15, 20, 3))
    const = 10
    for x in range(15):
        for y in range(20):
            img[x, y] = (255, 255, 255)
            if observation[x, y] == 5/const:
                #player
                img[x, y] = (0, 255, 0)
                continue
            if observation[x, y] == 10/const:
                #enemy
                img[x, y] = (0, 0, 255)
                continue
            if observation[x, y] == np.float16(7/const):
                #player&enemy
                img[x, y] = (0, 255, 255)
                continue
            if observation[x, y] == np.float16(1/const):
                #solid block
                img[x, y] = (0, 0, 0)
                continue

    dim = (img.shape[1] * 16, img.shape[0] * 16)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    cv.imshow('Computer Vision', img)
    # print('Remember to take care of this if statement. cv.destroyAllWindows does not need to be called in this very function')
    # if cv.waitKey(0) & 0xFF == ord('q'):
    #     cv.destroyAllWindows()

#import os
#when testing with this main routine, remember to check the path of the terminal that's running this
if __name__ == '__main__':
    img = window_capture('Samurai Gunn')
    cv.imshow('Screenshot', img)
    map = Map(0, img)
    
    while True:
        obs = observation(map)        
        renderObs(obs)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows() 
            break
