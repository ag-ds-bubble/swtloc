# Author : Achintya Gupta

import subprocess
from multiprocessing import current_process
import numpy as np
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import imutils
from cv2 import cv2

#######################PROGRESS BAR#######################
def prog_bar(_iterable, itrlen, mwidth = 40, individual_mode = False,
            comp_symb = '#', incomp_symb ='.',
            arch_hold = '', block_str = '', comp_msg = '', mp_lock = True):
    """
    Function for the Progress Bar. Wrapper Generator around and interable (list or dict)
    which yields each component indicating the increment in the progress bar
    """

    unitColor = '\033[5;49m\033[5;40m'
    endColor = '\033[5;30m\033[0;0m'

    check1 = isinstance(_iterable, list)
    check2 = isinstance(_iterable, dict)

    if check2:
        _iterable = _iterable.items()
    
    # Dont know why it works but the prog_bar ANSI codes wont work in
    # Windows otherwise. Need to figure this out.
    subprocess.call('', shell=True)


    if check1+check2 == 1:
        for idx, element in enumerate(_iterable):
            pct = (idx)/(itrlen)
            done = int(pct*mwidth)
            remain = mwidth - done
            msg = f'{idx}/{itrlen} Images Done'+"."*int((idx%5)+1)

            prog = '%s%s%s%s' % (unitColor, '\033[7m' + ' '*done + ' \033[27m', endColor, ' '*remain)
            prog_txt = "\r{0} @ {1} |{2}| -> STATUS: {3}% {4}".format(current_process().name, arch_hold, prog, np.round(pct*100, 1), msg)
            sys.stdout.write(prog_txt)
            sys.stdout.flush()
            import time
            time.sleep(0.01)
            yield element

        msg = f'{itrlen}/{itrlen} Images Done'+". Transformations Complete"
            
        prog = '%s%s%s' % (unitColor, '\033[7m' + ' '*int(mwidth/2 - 4) + 'COMPLETE' + ' '*int(mwidth/2 -4) + ' \033[27m', endColor)
        prog_txt = "\r{0} @ {1} |{2}| -> STATUS: {3}% {4}".format(current_process().name, arch_hold, prog, np.round(pct*100, 1), msg)
        sys.stdout.write(prog_txt)
        sys.stdout.flush()
        print('\n')
        
        
    else:
        raise TypeError("'prog_bar' can only accept one of \
                        ['list', 'dict'] as the iterable")



def auto_canny(image, sigma=0.33):
    """
    Autocanny Function
    Taken from : https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    Function to find Edge image from a grayscale image
    based on the thresholding parameter sigma.

    parameters
    --------------------------------------
    sigma : float, optional, default : 0.33

    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged    

def print_valcnts(image,_print=True, remove_0=True):
    """
    Calculate the value counts in a image
    """
    val, counts = np.unique(image, return_counts=True)
    vcdict = {k: v for k, v in sorted(dict(zip(val,counts)).items(), key=lambda item: item[1], reverse=True)}
    if remove_0:
        del vcdict[0]
    if _print:
        print("".join([str(k)+': '+str(v)+'\n' for k,v in vcdict.items()]))
    return vcdict

def prepCC(labelmask):
    """
    Prepare the Connected Components with 3 RGB channels
    """

    rmask = labelmask.copy()
    gmask = labelmask.copy()
    bmask = labelmask.copy()

    NUM_COLORS = len(np.unique(rmask))
    cm = plt.get_cmap('gist_rainbow')
    allcolors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    countdict = print_valcnts(labelmask, _print=False)
    
    for color, label in zip(allcolors, np.unique(rmask)):
        if label == 0:
            color = (0,0,0)
        rmask[rmask==label] = int(color[0]*255)
        gmask[gmask==label] = int(color[1]*255)
        bmask[bmask==label] = int(color[2]*255)

    colored_masks = np.dstack((rmask, gmask, bmask))
    return colored_masks.astype(np.uint8)

def imgshow(img, title='',imsize=(10,10)):
    """
    Show an image
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    
    fig,ax = plt.subplots(figsize=imsize)
    if len(img.shape) == 3:
        plt.imshow(imutils.convenience.opencv2matplotlib(img))
    elif len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

def imgsave(img, title, savepath, imsize=(10,10)):
    """
    Save an image
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    
    fig,ax = plt.subplots(figsize=imsize)
    if len(img.shape) == 3:
        plt.imshow(imutils.convenience.opencv2matplotlib(img))
    elif len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

def imgshowN(images:list, titles:list=[], place_pix_val=False,
             sup_title='Grouped Images', savepath=None, figsize = (10,10)):
    """
    Show N images
    """
    if titles == []:
        titles = ['Image '+str(k+1) for k in range(len(images))]
    if place_pix_val:
        for eimg in images:
            assert len(eimg.shape) == 2
    _rows = int(np.ceil(len(images)/3))
    _cols = len(images) if len(images)<=3 else 3
    
    fig = plt.figure(figsize=figsize, dpi = 100)
    # plt.xlabel(sup_title, fontsize=10) # TODO : Borders too wide
    # plt.tick_params(top=False, bottom=False, left=False, right=False,
    #                 labelleft=False, labelbottom=False)
    plt.rcParams['figure.dpi']=120

    grid = ImageGrid(fig, 111, nrows_ncols=(_rows, _cols), axes_pad=0.1)

    for _img , _title, _ax in zip(images, titles, grid):
        if _img.shape[-1] == 3:
            _ax.imshow(imutils.convenience.opencv2matplotlib(_img))
        else:
            _ax.imshow(_img, cmap='gray')
            
            if place_pix_val:
                for  _y in range(_img.shape[0]):
                    for _x in range(_img.shape[1]):
                        _ax.annotate(_img[_y,_x], (_x-0.3,_y), color='r', fontsize=6)
                        
        _ax.set_title(_title)
        
        
    for _delax in grid[len(images):]:
        fig.delaxes(_delax)
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()
    
def resize_maintinaAR(image, width=1.0, height=1.0, inter=cv2.INTER_AREA, mode='proportion'):
    """
    A function to resize the image based on the params.
    
    Arguments
    ------------------------------
        image : Original Image, np.ndarray
            Image to resize
        
        width(Optional) : int or float.
            How much to resize based on the width.
        
        height(Optional) : int or float
            How much to resize based on the height.
        
        inter(Optional) : opencv interpolation mode
        
        mode(Optional) : One of 'proportion' or 'actual'
            Which mode to resize the image in.
    Returns
    ------------------------------
    Resized image
    """
    #Adopted from : https://stackoverflow.com/a/55306956/6297658
    dim = None
    (h, w) = image.shape[:2]

    if mode == 'proportion':
        width *= w
        width = int(width)
        height *= h
        height = int(height)

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

