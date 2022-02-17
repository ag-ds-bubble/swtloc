# Author : Achintya Gupta
# Purpose : Houses utility functions

import functools
import numpy as np
from cv2 import cv2
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

CEND = '\33[0m'
CBOLD = '\33[1m'
CREDBG = '\033[41m'


def print_in_red(text: str) -> None:
    """
    Print a red color text.
    Args:
        text (str) : Text to print
    Example:
    ::
        >>> # Printing a red text
        >>> print_in_red('This is a red text')
        This is a red text
    """
    print(CREDBG + text + CEND)


def perform_type_sanity_checks(cfg: Dict, cfg_of: str) -> None:
    """
    Perform type sanity checks for the values provided in `cfg` for a particular
    `cfg_of` configuration type.
    Args:
        cfg (dict) : Configuration dictionary
        cfg_of (str) : Configuration of which Type Sanity Checks need to be performed
    Raises:
        SWTTypeError, SWTValueError
    """
    truncated_cfg = {k: v for k, v in cfg.items() if cfg_of in k}
    all_keys = list(set([k.split('.')[-1] for k, v in truncated_cfg.items()]))
    all_keys = [k for k in all_keys if k not in ['options', 'type']]

    for each_cfg_key in all_keys:
        cfg_val = truncated_cfg.get(f'{cfg_of}.{each_cfg_key}')
        cfg_types = truncated_cfg.get(f'{cfg_of}.{each_cfg_key}.type')
        cfg_options = truncated_cfg.get(f'{cfg_of}.{each_cfg_key}.options')
        param_name = each_cfg_key.split('.')[-1]

        cfg_val_type_check = [isinstance(cfg_val, k) for k in cfg_types]
        if not (cfg_val_type_check.count(True) == 1):
            raise SWTTypeError(f"`{param_name}` value should be one of these types : {cfg_types}. Not mixed either.")

        if cfg_options:
            if cfg_val not in cfg_options:
                raise SWTValueError(f"`{param_name}` can assume only one of {cfg_options}. `{cfg_val}` was given.")


def deprecated_wrapper(reason: str, in_favour_of: str, removed_in: str, relocated: Optional[bool] = False):
    """
    A decorator to mark the deprecated functions and give the reason
    of deprecation. Alongside the reason, also inform in which version
    will the function be removed. If the function is being relocated then also
    inform in favour of which function will this function will be relocated
    Args:
        reason (str) : Reason for relocation/deprecation
        in_favour_of (str) : If relocated, in favour of which function is this
         function being relocated
        removed_in (str) : In which version will the function be removed
        relocated (Optional[bool]) : Is the deprecation for being relocated
    """

    if not relocated:
        warn_message = "DEPRECATED : `{fname}` function deprecated in favour of {in_favour_of}. "
        warn_message += "Reason - {reason}; To be Removed in - {removed_in}"
    else:
        warn_message = "RELOCATED : `{fname}` function relocated to {in_favour_of}. "
        warn_message += "Reason - {reason}; To be Removed in - {removed_in}"

    def deprecated_function(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            print_in_red(text=warn_message.format(fname=func.__name__, in_favour_of=in_favour_of,
                                                  reason=reason, removed_in=removed_in))
            return func(*args, **kwargs)

        return new_func

    return deprecated_function


def generate_random_swtimage_names(n: int) -> List[str]:
    """
    Generates random image names made of random numbers
    Args:
        n (int) : Number of names to generate
    Returns:
        (List[str]) : List of string names to generate.
    Example:
    ::
        >>> # Generating `n` random integer string (names)
        >>> generate_random_swtimage_names(3)
        ['SWTImage_982112', 'SWTImage_571388', 'SWTImage_866821']
    """
    random_names: List[str] = []
    np.random.seed(999)
    for _ in range(n):
        random_names.append(f'SWTImage_{np.random.randint(100_000, 999_999)}')
    return random_names


def auto_canny(img: np.ndarray, sigma: Optional[float] = 0.33) -> np.ndarray:
    """
    Autocanny Function.
     Taken from : https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
     Function to find Edge image from a grayscale image
     based on the thresholding parameter sigma.
    Args:
        img (np.ndarray) : Input gray-scale image.
        sigma (Optional[float]): Sigma Value,default : 0.33
    Example:
    ::
        >>> # Generating Canny Edge Image
        >>> root_path = '../swtloc/examples/test_images/'
        >>> single_image_path = root_path+'test_img1.jpg'
        >>> original_image = cv2.imread(single_image_path)
        >>> edge_image = auto_canny(img=original_image, sigma=0.2)
        >>> print(original_image.shape, edge_image.shape)
        (768, 1024, 3) (768, 1024)
    """
    # Compute the median of the single channel pixel intensities
    v = np.median(img)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)

    return edged


def image_1C_to_3C(image_1C: np.ndarray, all_colors: Optional[List[Tuple[int]]] = None,
                   scale_with_values: Optional[bool] = False) -> np.ndarray:
    """
    Prepare the RGB channel image from a single channel image (gray-scale).
    Each unique integer in `image_1C` will be given a unique color, unless
    `all_colors` parameter is provided. `scale_with_values` parameter ensures
    color so generated (if `all_colors` parameter not given) will be generated
    using the *sequential* matplotlib color scheme.

    Args:
        image_1C (np.ndarray) : Input single channel image, which needs to be transformed
        all_colors (Optional[List[Tuple[int]]]) : Colors corresponding to each unique integer in the image
        scale_with_values (Optional[bool]) : Whether to use matplotlib *sequential* color map or not.
    Returns:
        (np.ndarray) : Three channel image, after the conversions of the single channel image
    """
    rmask = image_1C.copy()
    gmask = image_1C.copy()
    bmask = image_1C.copy()

    unique_labels = np.unique(rmask)
    num_colors = len(unique_labels)

    if not all_colors:
        if scale_with_values:
            cm = plt.get_cmap('Oranges')
        else:
            cm = plt.get_cmap('tab20c')
        all_colors = [cm(1. * i / num_colors) for i in range(num_colors)]

    for color, label in zip(all_colors, sorted(unique_labels)):
        mask_indices = np.where(rmask == label)
        if label == 0:
            color = (0, 0, 0)

        rmask[mask_indices] = int(color[2] * 255)
        gmask[mask_indices] = int(color[1] * 255)
        bmask[mask_indices] = int(color[0] * 255)

    colored_masks = np.dstack((rmask, gmask, bmask))
    return colored_masks.astype(np.uint8)


def show_N_images(images: List[np.ndarray],
                  individual_titles: List[str],
                  plot_title: Optional[str] = 'SWTLoc Plot',
                  sup_title: Optional[str] = '',
                  return_img: Optional[bool] = False) -> Optional[np.ndarray]:
    """
    Display n (<=4) images in a grid.

    Args:
        images (List[np.ndarray]) : List of images to display
        individual_titles (List[str]) : Title for each image to be displayed
        plot_title (Optional[str]) : Plot Title. [default = 'SWTLoc Plot']
        sup_title (Optional[str]) : Plot sub title. [default = '']
        return_img (Optional[bool]) : Whether to return the plotted figure or not. [default = False]
    Raises:
        SWTValueError
    Returns:
        (matplotlib.figure.Figure) : Plotted figure if the `return_fig` parameter was given as `True`
    """
    nimages = len(images)
    if nimages <= 3:
        _rows = 1
        _cols = nimages
        fig_size = (12, 5.5)
    elif nimages == 4:
        _rows = 2
        _cols = 2
        fig_size = (12, 9)
    else:
        raise SWTValueError("Maximum of 4 images allowed")

    fig = plt.figure(figsize=fig_size, dpi=98)
    plt.suptitle(plot_title + sup_title, fontsize=14)

    grid = ImageGrid(fig, 111, nrows_ncols=(_rows, _cols), axes_pad=0.4, label_mode='L')

    for _img, _title, _ax in zip(images, individual_titles, grid):
        if _img.shape[-1] == 3:
            _ax.imshow(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
        else:
            _ax.imshow(_img, cmap='gray')

        _ax.set_title(_title, fontsize=10)

    for _delax in grid[len(images):]:
        fig.delaxes(_delax)

    if return_img:
        return fig

    plt.show()


def unique_value_counts(image: np.ndarray, remove_0: Optional[bool] = True) -> Dict:
    """
    Calculate unique integers in an image and their counts

    Args:
        image (np.ndarray) : Image of which the unique values need to be calculated
        remove_0 (Optional[bool]) : Whether to remove integer 0 from the calculated
         dictionary or not.
    Returns:
        (dict) : Dictionary containing key as unique integer and value as counts
         of that integer in the image
    """
    val, counts = np.unique(image, return_counts=True)
    vcdict = {k: v for k, v in sorted(dict(zip(val, counts)).items(), key=lambda item: item[1], reverse=True)}
    if remove_0:
        vcdict.pop(0, None)
    return vcdict


def get_connected_components_with_stats(img: np.ndarray):
    """
    Function to find the connected components alongside their
    stats using oepncv's connectedComponentWithStats function for
    any given image.

    Args:
        img (np.ndarray) : Input Image
    Returns:
        (Tuple[int, np.ndarray, np.ndarray, np.ndarray]) : Results of opencv connectedComponentsWithStats
    """
    threshmask = img.copy().astype(np.int16)
    threshmask[threshmask > 0] = 1
    threshmask = threshmask.astype(np.int8)
    return cv2.connectedComponentsWithStats(threshmask, connectivity=8)


class SWTLocExceptions(Exception):
    """Base class for SWTLoc Exceptions"""


class SWTLocalizerValueError(SWTLocExceptions):
    """Raised when wrong input is provided to the `SWTLocalizer` class"""


class SWTImageProcessError(SWTLocExceptions):
    """Raised when functions don't follow a proper flow for SWTImage"""


class SWTValueError(SWTLocExceptions):
    """Raised when non-acceptable value is received in the parameters"""


class SWTTypeError(SWTLocExceptions):
    """Raised when non-acceptable type is received in the parameters"""
