import numpy as np

# pywin32 no funciona con python 3.9*
# Para restaurar compatibilidad, un parche fue aplicado.
# Source: https://stackoverflow.com/questions/64655079/cannot-import-from-pywinauto-importerror-dll-load-failed-while-importing-win32

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

def window_capture(window_name, save=False, bmpFileName='sample.jpg'):
    '''
    Takes a screenshot of the given window. 
    Note that this implementation is tuned with a couple hard-coded values to match the 
    windowed display for the game "Samurai Gunn"

    Parameters:
        window_name: (string) name of the window that needs capturing.
        save: (boolean) answers the question: Do you want to save the screenshot to a file? 
        bmpFileName: (string) name given to the file when saving the screenshot. Must include extension.

    Returns:
        img: (ndarray) Array of dimansions (h=240, w=320, 4). The 4th channel is the alpha value.
    '''
    hwnd = win32gui.FindWindow(None, window_name)
    
    # Getting the window's size and accounting for window screenshot borders
    # does not work consistently?
    #window_rect = win32gui.GetWindowRect(hwnd)
    
    # dimensions of titlebar_px=31 and border_px=8 for fer's computer
    # dimensions of titlebar_px=38 and border_px=9 for Pablo's pc

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
    if save:
        dataBitMap.SaveBitmapFile(cDC, bmpFileName)
    
    # Converting to format useful for opencv
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (h, w, 4)
    
    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    
    # Dropping alpha channel may be useful for some applications, like when using
    # cv.matchTemplate(), which may throw an error otherwise
    
    #img = img[...,:3]   # this drops alpha channel. 
    
    return img

if __name__ == "__main__":
    window_capture('Samurai Gunn', save=True, bmpFileName='test.jpg')
