import cv2 as cv
import numpy as np
from time import time
import win32gui
import win32ui
import win32con

def list_all_open_windows():
    '''Lists the names and handles of all opened windows
    '''
    def winEnumHandler( hwnd, ctx ):
        if win32gui.IsWindowVisible( hwnd ):
            print (hex(hwnd), win32gui.GetWindowText( hwnd ))

    win32gui.EnumWindows(winEnumHandler, None)

# list_all_open_windows()

def window_capture():
    hwnd = win32gui.FindWindow(None, 'Samurai Gunn')
    
    # Getting the window's size and accounting for window screenshot borders
    # does not work consistently?
    #window_rect = win32gui.GetWindowRect(hwnd)
    
    # dimensions of titlebar=31 and border=8 for fer's computer
    # para notebook de Pablo titlebar_px = 38,  border_px = 9

    titlebar_px = 38
    border_px = 9
    
    # For samurai Gunn, the non-fullscreen dimensions should be:
    w = 320
    h = 240
    
    crop_x = border_px
    crop_y = titlebar_px
    
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h) , dcObj, (crop_x, crop_y), win32con.SRCCOPY)
    
    # To save screenshot to file, uncomment the 2 lines below 
    # bmpfilenamename = "prueba.jpg" #set this
    # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
    
    # Converting to format useful for opencv
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (h, w, 4)
    
    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    
    # Dropping alpha channel may be useful for some applications, like cv.matchTemplate()
    # which may throw an error otherwise
    
    #img = img[...,:3]   # this drops alpha channel 
    
    return img

row_step = 2
col_step = 2
def classify_tile(x, y, img, sliced):
    '''
    Tiles are (16 x 16)px
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
    px_count = 0
    total = np.zeros((1, img.shape[2]), dtype='uint32')
    
    corrected_height_range = int(16/(1+sliced))
    for i in range(0, corrected_height_range, row_step):
        for j in range(0, 16, col_step):
            total += img[x+i][y+j] # [b, g, r, a]
            px_count += 1           
    
    #result = result.astype('uint8') 
    # for some reason this doesn't work. It's fine tho, it works if
    # we convert the data type before returning on the simplify() funct
    result = total * ((1+sliced)/(px_count))
    result = result[:3]
    
    return result

def lives(img):
    '''
    We settled a standar player for the agent which is the player 1 and the green character. 
    His lives are static in the top left area of the map. For now this part of the coding seek
    a pair of coordinates on the image matrix that will represent each of the ten lives for the green player, 
    so the agent know the lives he has at every moment. It also works for the rewarding in future plans. 

    Coordinates of the 10 lives at the distance of 3 pixels high and 3 pixels width
    from the first pixel on the top left side of the "leaf" are:

    (7,8)       (7,15)
    (14,8)      (14,15)
    (21,8)      (21,15)
    (28,8)      (28,15)
    (35,8)      (35,15)

   # Hay un problema porque hay que recordar que hay imagenes que van a tener medio bloque 
   # de distancia  y esos son 8 pixeles creo. 

    For green characters leafs are on 208 in green intensity
    '''
    img_without_alpha= img[...,:3]
    b_mapa,g_mapa,r_mapa = cv.split(img_without_alpha)

    life = 0      #contador de vidas

    for fila in range (7,36,7):
        for col in range (8,16,7):
            intensity= g_mapa[fila,col]

            if intensity == 208:
                life+=1
    return(life)
        # Aquí el else deberia ser una respuesta negativa, como un reward negativo (?)"
        
def enemy_lives(img):
    '''
    For finding the enemy lives, we used the cv.matchTemplate function which review the template in the corresponding image
    but in gray scale, which was previously converted. The template was created manually with black background and 
    the white skull, that represents the lives of the enemy in survival mode. We reduced the area were the template
    was evaluated from the entire image to the section were the lives of the enemy stayed static. For avoiding 
    multiple detection the threshold for the mathcTemplate was 0.6 and determined arbitray according to various scenarios.     
    '''
    gray_img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    count_enemy=0
    template= cv.imread("template_mini.bmp",0)     
    line=np.zeros((20,80),dtype='uint8')
    for fila in range(0,20):
        for col in range(0,80):
            line[fila,col]=gray_img[fila+215, col+120]
    enemy_lv= np.zeros((20,80),dtype='uint8')    
    locations= cv.matchTemplate(line, template, cv.TM_CCOEFF_NORMED)
    thresh=0.6
    locations_w= np.where(locations >= thresh)
    locations_f= list(zip(*locations_w[::-1]))    
    count_enemy=len(locations_f)    
   # The section below shows a window with the section of the screenshot with the lives of the enemy (which is a skull) 
   # and a rectangle that surrounds the skull.
   # if locations_f:
   #     needle_w= template.shape[1]
   #     needle_h= template.shape[0]
   #     line_color=(255,0,0)
   #     line_type=cv.LINE_4
   #     # loop over all locations and draw their rectangle
   #     for loc in locations_f:
   #         #box position
   #         top_left=loc
   #         bottom_right= (top_left[0] + needle_w, top_left[1] + needle_h)
            #draw box
   #         enemy_lv=cv.rectangle(line,top_left, bottom_right, line_color, line_type)
   # else:
   #     print('no se llama')        
    print('Enemy lives:', count_enemy)    
    return(enemy_lv)

def map_mask(img):
    mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    size = mask.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if mask[i][j] < 70 or mask[i][j] > 230:
                mask[i][j] = 1
            elif mask[i][j] > 100 and mask[i][j] < 190:
                mask[i][j] = 0
    return mask

def simplify(img, offset, mask = None):
    '''Reduces the input image resolution by classifying the tiles on the screen and 
    reducing them to 1 px per tile.
    The tiles in every map can be aligned with a 20 x 15 grid of (16 x 16)px cells.
    Iterates over each tile, calling classify_tile() for each of them
    
    Parameters:
    img: input image
    offset: in case the map tiles do not align perfectly with the grid, an offset of 8px down does 
            the trick. (Maps with different offsets have yet to be found, 
            this should allow for different offsets in the X direction)
            
    Returns:
    numpy 3D array: (15 x 20 x num_channels) numpy array
    '''
    sliced = 0
    simple_img = np.zeros((15, 20, img.shape[2]))    
    for x in range(15):
        if offset == 8 and (x == 0 or x == 19):
            sliced = 1
        else:
            sliced = 0
        for y in range(20):
            # pass the coordinates of each tile with corrections to accomodate for the grid offset
            if mask is None or mask[x][y] == 0:
                simple_img[x, y] = classify_tile_2(x*16 - (offset*(1-sliced)), 
                                                   y*16 - (offset*(1-sliced)), 
                                                   img, sliced)
    simple_img = simple_img.astype('uint8')
    return simple_img

def simplify_2(img, offset, mask = None):
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
    # clasifica segun la moda.
    # parece funcionar bien para mapear plataformas
    row_step = 2
    col_step = 1
    freq = {}
    corrected_height_range = int(16/(1+sliced))
    for i in range(0, corrected_height_range, row_step):
        for j in range(0, 16, col_step):
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

player_temp_r = cv.imread('char_templates/splinter_needle_r.jpg')
player_temp_l = cv.imread('char_templates/splinter_needle_l.jpg')
enemy_temp = cv.imread('char_templates/ninja_needle.jpg')

methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# "player key color" this is a distinctive color for the rat-like player
# funciona en BGR (???)
pkc = (65, 191, 97)
# "enemy key color" this is a distinctive color for the ninja enemy bot in survival mode
ekc = (11, 11, 13)


def find_players(x, y, img, sliced):
    row_step = 2
    col_step = 2   
    tile = np.zeros((16, 16, 3), dtype='uint8')
    
    p_threshold = 0.6
    e_threshold = 0.312
    
    player_flag = False
    enemy_flag = False
    corrected_height_range = int(16/(1+sliced))
    for i in range(0, corrected_height_range, row_step):
        for j in range(0, 16, col_step):
            aux = img[x+i][y+j][:3]
            if tuple(aux) == pkc:
                player_flag = True
                #print('player flag set')
            if tuple(aux) == ekc:
                enemy_flag = True
                #print('enemy flag set')
                
    if player_flag or enemy_flag:
        for i in range(0, corrected_height_range):
            for j in range(0, 16):
                tile[i][j] = img[x+i][y+j][:3]
            
                
    min_val_p, max_val_p, min_loc_p, max_loc_p = (0, 0, (0,0), (0,0))
    min_val_e, max_val_e, min_loc_e, max_loc_e = (0, 0, (0,0), (0,0))
    
    if player_flag:
        search_p1 = cv.matchTemplate(tile, player_temp_r, cv.TM_CCOEFF_NORMED)
        min_val_p1, max_val_p1, min_loc_p, max_loc_p = cv.minMaxLoc(search_p1)
        
        search_p2 = cv.matchTemplate(tile, player_temp_l, cv.TM_CCOEFF_NORMED)
        min_val_p2, max_val_p2, min_loc_p, max_loc_p = cv.minMaxLoc(search_p2)
        
        max_val_p = max_val_p1 if (max_val_p1 > max_val_p2) else max_val_p2
        # min_val_p = min_val_p1 if (min_val_p1 < min_val_p2) else min_val_p2
    if enemy_flag:
        search_e = cv.matchTemplate(tile, enemy_temp, cv.TM_CCOEFF_NORMED)
        min_val_e, max_val_e, min_loc_e, max_loc_e = cv.minMaxLoc(search_e)
        # print('max_val for enemy {}'.format(max_val_e))
    result = np.zeros(3)
    if max_val_p > p_threshold and max_val_e < e_threshold:
        return pkc
    elif max_val_p < p_threshold and max_val_e > e_threshold:
        #return ekc
        return (0, 0, 255)
    elif max_val_p > p_threshold and max_val_e > e_threshold:
        avg = np.zeros(3, dtype='uint8')
        for i in range(3):
            avg[i] = (pkc[i] + ekc[i]) / 2
        return tuple(avg)
    else:
        return (255, 255, 255)
  

if __name__ == "__main__":      
    frame_count = 1
    cumulative_fps = 0
    ss_counter = 0  # "screenshot counter"
    
    # offset will depend on the map
    # for Ice cube, offset is 0
    offset = 0
    mk = None
    while(True):
        prev_time = time()
        
        screenshot = window_capture()
        #screenshot = cv.imread('sample0.jpg')
        
        img = simplify_2(screenshot, offset, mask=mk)
        
        # re-calculate mask every so often
        if frame_count % 128 == 0: 
            mk = map_mask(simplify_2(screenshot, offset))
        
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
        if frame_count % 512 == 0:
            print("Player lives:",  pj_lives)
            
            enemy = enemy_lives(screenshot)
            cv.imshow('Area of enemy lives', enemy)
            frame_count = 1
            cumulative_fps = 0
        
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
    
