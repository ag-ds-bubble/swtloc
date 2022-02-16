# Author : Achintya Gupta
# Purpose : Houses Abstractions for Stroke Width Transforms

import math
import os
import time
from copy import deepcopy
from typing import ByteString
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2

from .base import GroupedComponentsBase
from .base import IndividualComponentBase
from .base import TextTransformBase
from .configs import CODE_VAR_NAME_MAPPINGS
from .configs import (CONFIG__SWTIMAGE__GETLETTER,
                      CONFIG__SWTIMAGE__GETLETTER_KEY,
                      CONFIG__SWTIMAGE__GETLETTER_LOCALIZE_BY,
                      CONFIG__SWTIMAGE__GETLETTER_DISPLAY)
from .configs import (CONFIG__SWTIMAGE__GETWORD,
                      CONFIG__SWTIMAGE__GETWORD_KEY,
                      CONFIG__SWTIMAGE__GETWORD_LOCALIZE_BY,
                      CONFIG__SWTIMAGE__GETWORD_DISPLAY)
from .configs import (CONFIG__SWTIMAGE__LOCALIZELETTERS,
                      CONFIG__SWTIMAGE__LOCALIZELETTERS_MAXIMUM_PIXELS_PER_CC,
                      CONFIG__SWTIMAGE__LOCALIZELETTERS_MINIMUM_PIXELS_PER_CC,
                      CONFIG__SWTIMAGE__LOCALIZELETTERS_ACCEPTABLE_ASPECT_RATIO,
                      CONFIG__SWTIMAGE__LOCALIZELETTERS_LOCALIZE_BY,
                      CONFIG__SWTIMAGE__LOCALIZELETTERS_PADDING_PCT,
                      CONFIG__SWTIMAGE__LOCALIZELETTERS_DISPLAY)
from .configs import (CONFIG__SWTIMAGE__LOCALIZEWORDS,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_LOCALIZE_BY,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_LOOKUP_RADIUS_MULTIPLIER,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_STROKE_WIDTH_RATIO,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_COLOR_DEVIATION,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_HEIGHT_RATIO,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_ANGLE_DEVIATION,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_POLYGON_DILATE_ITERATIONS,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_POLYGON_DILATE_KERNEL,
                      CONFIG__SWTIMAGE__LOCALIZEWORDS_DISPLAY)
from .configs import (CONFIG__SWTIMAGE__SAVECROPS,
                      CONFIG__SWTIMAGE__SAVECROPS_SAVE_PATH,
                      CONFIG__SWTIMAGE__SAVECROPS_CROP_OF,
                      CONFIG__SWTIMAGE__SAVECROPS_CROP_KEY,
                      CONFIG__SWTIMAGE__SAVECROPS_CROP_ON)
from .configs import (CONFIG__SWTIMAGE__SHOWIMAGE,
                      CONFIG__SWTIMAGE__SHOWIMAGE_IMAGE_CODES,
                      CONFIG__SWTIMAGE__SHOWIMAGE_PLOT_TITLE,
                      CONFIG__SWTIMAGE__SHOWIMAGE_PLOT_SUP_TITLE)
from .configs import (CONFIG__SWTIMAGE__TRANSFORM,
                      CONFIG__SWTIMAGE__TRANSFORM_GAUSSIAN_BLURR,
                      CONFIG__SWTIMAGE__TRANSFORM_GAUSSIAN_BLURR_KERNEL,
                      CONFIG__SWTIMAGE__TRANSFORM_EDGE_FUNCTION,
                      CONFIG__SWTIMAGE__TRANSFORM_AUTO_CANNY_SIGMA,
                      CONFIG__SWTIMAGE__TRANSFORM_MAXIMUM_ANGLE_DEVIATION,
                      CONFIG__SWTIMAGE__TRANSFORM_MINIMUM_STROKE_WIDTH,
                      CONFIG__SWTIMAGE__TRANSFORM_MAXIMUM_STROKE_WIDTH,
                      CONFIG__SWTIMAGE__TRANSFORM_TEXT_MODE,
                      CONFIG__SWTIMAGE__TRANSFORM_CHECK_ANGLE_DEVIATION,
                      CONFIG__SWTIMAGE__TRANSFORM_ENGINE,
                      CONFIG__SWTIMAGE__TRANSFORM_INCLUDE_EDGES_IN_SWT,
                      CONFIG__SWTIMAGE__TRANSFORM_DISPLAY)
# Image Codes
from .configs import (IMAGE_CONNECTED_COMPONENTS_1C,
                      IMAGE_CONNECTED_COMPONENTS_3C,
                      IMAGE_CONNECTED_COMPONENTS_PRUNED_1C,
                      IMAGE_CONNECTED_COMPONENTS_PRUNED_3C,
                      IMAGE_CONNECTED_COMPONENTS_3C_WITH_PRUNED_ELEMENTS)
from .configs import (IMAGE_INDIVIDUAL_LETTER_LOCALIZATION,
                      IMAGE_ORIGINAL_INDIVIDUAL_LETTER_LOCALIZATION,
                      IMAGE_INDIVIDUAL_WORD_LOCALIZATION,
                      IMAGE_ORIGINAL_INDIVIDUAL_WORD_LOCALIZATION,
                      CONFIG__SWTIMAGE__SHOWIMAGE_SAVE_DIR,
                      CONFIG__SWTIMAGE__SHOWIMAGE_SAVE_FIG,
                      CONFIG__SWTIMAGE__SHOWIMAGE_DPI)
from .configs import (IMAGE_ORIGINAL,
                      IMAGE_GRAYSCALE,
                      IMAGE_EDGED,
                      IMAGE_SWT_TRANSFORMED)
from .configs import (TRANSFORM_INPUT__3C_IMAGE,
                      IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS,
                      IMAGE_ORIGINAL_LETTER_LOCALIZATIONS,
                      IMAGE_ORIGINAL_MASKED_LETTER_LOCALIZATIONS,
                      IMAGE_PRUNED_3C_WORD_LOCALIZATIONS,
                      IMAGE_ORIGINAL_WORD_LOCALIZATIONS,
                      IMAGE_ORIGINAL_MASKED_WORD_LOCALIZATIONS)
from .configs import get_code_descriptions
from .core import Fusion
from .core import ProxyLetter
from .core import swt_strokes
from .core import swt_strokes_jitted
from .utils import SWTImageProcessError
from .utils import SWTValueError
from .utils import auto_canny
from .utils import image_1C_to_3C
from .utils import perform_type_sanity_checks
from .utils import get_connected_components_with_stats
from .utils import print_in_red
from .utils import show_N_images
from .utils import unique_value_counts

_LETTER_SUP_TITLE_MAPPINGS = {"outline": "Outline",
                              "ext_bbox": "External\ Bounding\ Box",
                              "min_bbox": "Minimum\ Bounding\ Box"}
_WORD_SUP_TITLE_MAPPINGS = {"polygon": "Polygon",
                            "bbox": "Bounding\ Box",
                            "bubble": "Bubble"}


class Letter(IndividualComponentBase):
    """
    ``Letter`` class represents, a letter - an individual component
    which houses various properties of that individual letter.
    """

    def __init__(self, label: int, image_height: int, image_width: int):
        """
        Create an ``Letter`` object which will house the components
        properties such as :
            - Minimum Bounding Box & its related properties
            - External Bounding Box & its related properties
            - Outline (Contour)
            - Original Image Properties
            - Stroke Width Properties
        Args:
            label (int) : A unique identifier for this Component
            image_height (int) : Height of the image in which this component resides
            image_width (int) : Width of the image in which this component resides
        """
        super().__init__(label, image_height, image_width)
        # Mean Stroke Widths in this Letter
        self.stroke_widths_mean: float = 0.0
        # Unique Stroke Widths and their counts
        self.stroke_widths_counts: dict = dict()
        # Median Stroke Widths in this Letter
        self.stroke_widths_median: float = 0.0
        # Variance of Stroke Widths in this Letter
        self.stroke_widths_variance: float = 0.0

    def __repr__(self):
        """Representational String"""
        return f"Letter-{self.label}"

    def _setLetterProps(self, area: int, sw_mean: np.ndarray, sw_median: np.ndarray, sw_var: np.ndarray,
                        sw_counts: Dict[int, int], color_mean: np.ndarray, color_median: np.ndarray,
                        outline: Union[List, np.ndarray]):
        """
        Set Letter properties corresponding to
            - Area (Pixels) of the letter (Individual Letter)
            - Mean Stroke Width of the letter (Individual Letter)
            - Median Stroke Width of the letter (Individual Letter)
            - Variance of Stroke Width of the letter (Individual Letter)
            - Mean color of the letter (Individual Letter)
            - Median color of the letter (Individual Letter)

        Args:
            area (int) : Area (Pixels) of the letter. (Attribute : `area_pixels`)
            sw_mean (np.ndarray) : Mean Stroke Width of the letter. (Attribute : `stroke_widths_mean`)
            sw_median (np.ndarray) : Median Stroke Width of the letter. (Attribute : `stroke_widths_median`)
            sw_var (np.ndarray) : Variance Stroke Width of the letter. (Attribute : `stroke_widths_variance`)
            sw_counts (Dict[int, int]) : Dictionary containing the mapping of various strokes in this
             letter and their counts. (Attribute : `stroke_widths_counts`)
            color_mean (np.ndarray) : Mean color of the letter in the original image, across channels. (Attribute : `original_color_mean`)
            color_median (np.ndarray) : Median color of the letter in the original image, across channels. (Attribute : `original_color_median`)
            outline (np.ndarray) : Outline (Contour)of the letter in the original image. (Attribute : `outline`)
        """
        self._setIcProps(area=area, color_mean=color_mean, color_median=color_median, outline=outline)
        self.stroke_widths_mean = sw_mean
        self.stroke_widths_median = sw_median
        self.stroke_widths_variance = sw_var
        self.stroke_widths_counts = sw_counts

    def _checkAvailability(self, localize_by: str):
        """
        Check if properties for a particular `localize_by` are available and populated.

        Args:
            localize_by (str) : Which localization to check the properties availability of
                - `min_bbox` : Minimum Bounding Box
                - `ext_bbox` : External Bounding Box
                - `outline` : Contour
                - `circular` : Circle - With Minimum Bounding Box Centre coordinate and radius
                  = Minimum Bounding Box Circum Radius * radius_multiplier
        Raise:
             SWTImageProcessError
        """
        temp_img = getattr(self, localize_by)
        if np.array(temp_img).size == 0:
            raise SWTImageProcessError(
                f"'SWTImage.localizeLetters' with localize_by='{localize_by}' should be run before this.")


class Word(GroupedComponentsBase):
    """
    ``Word`` class represents, a word - connected component
    which houses various properties of that individual word.
    """

    def __init__(self, label: int, letters: List[Letter], image_height: int, image_width: int):
        """
        Create an ``Word`` object which will house the grouped components
        properties such as :
            - Various Bounding Shapes which house that particular grouped component entirely
        Args:
            letters (List[Letter]) : Letters which can be grouped into this word.
            label (int) : A unique identifier for this Component
            image_height (int) : Image height
            image_width (int) : Image Width
        """
        super().__init__(label, image_height, image_width)
        # Letters in this Word
        self.letters: List[Letter] = letters
        # Labels of the Letters in this Word
        self.letter_labels: List[int] = [each_letter.label for each_letter in letters]
        # Number of Letters in this Word
        self.nletters: int = len(self.letters)

    def __repr__(self):
        """Representational String"""
        return f"Word-{self.label}"

    def _checkAvailability(self, localize_by):
        """
        Check if properties for a particular `localize_by` are available and populated.

        Args:
            localize_by (str) : Which localization to check the properties availability of
                - `bbox` : Bounding Box
                - `bubble` : Bubble Boundary
                - `polygon` : Contour Boundary
        Raises:
             SWTImageProcessError
        """
        temp_img = getattr(self, localize_by)
        if np.array(temp_img).size < 2:
            raise SWTImageProcessError(
                f"'SWTImage.localizeWords' with localize_by='{localize_by}' should be run before this.")


class SWTImage(TextTransformBase):
    """
    This class houses the procedures for
        - Transforming
        - Localizing Letters
        - Localizing Words

    Objects of this class are made and stored in ``SWTLocalizer`` class attribute `swtimages`

    This class serves as an abstraction to various operations that can be performed via transforming
    the image through the Stroke Width Transform. This class also includes helper functions to extend
    the ability to save, show and crop various localizations and intermediary stages as well.
    """

    def __init__(self, image: np.ndarray, image_name: str, input_flag: ByteString, cfg: Dict):
        """
        Create an ``SWTImage``, an abstraction to various procedures to be performed on a ***single***
        input image.
        
        Args:
            image (np.ndarray) : Input image on which transformation will be performed

            image_name (str) : Name of the input images (Needed while saving the post-transformation results)

            input_flag (ByteString) : Flag of input type. It can be only one of the following
                - `TRANSFORM_INPUT__1C_IMAGE` = b'21'
                - `TRANSFORM_INPUT__3C_IMAGE` = b'22'
                These image codes reside in configs.py file

            cfg (dict) : Configuration of a particular transformation type.
        """
        super().__init__(image, image_name, input_flag, cfg)
        # > Parameters for transformImage
        self.image_grayscale: np.ndarray = np.array([])  # Stage-1
        self.image_gaussian_blurred: np.ndarray = np.array([])  # Stage-2
        self.image_edged: np.ndarray = np.array([])  # Stage-3
        self.image_gradient_theta: np.ndarray = np.array([])  # Stage-4
        self.hstep_mat: np.ndarray = np.array([])  # Stage-5a
        self.vstep_mat: np.ndarray = np.array([])  # Stage-5b
        self.dstep_mat: np.ndarray = np.array([])  # Stage-5c
        self.image_swt: np.ndarray = np.array([])  # Stage-6

        # > Parameters for localizing letters
        self.letters: Dict[int, Letter] = dict()

        # > Parameters for localizing words
        self.words: Dict[int, Word] = dict()

    def __repr__(self):
        return f"SWTImage-{self.image_name}"

    # ######################################### #
    #                  TRANSFORM                #
    # ######################################### #

    def _resetSWTTransformParams(self):
        """
        Resets the Transform stage parameters and the downstream stage parameters :
            - findAndPrune Parameters
            - localizeLetters Parameters
            - localizeWords Parameters
        Alongside them, attributes pertaining to Stroke Width Transforms are also reset.
        """
        self._resetTransformParams()
        self.letters: Dict[int, Letter] = dict()
        self.words: Dict[int, Word] = dict()
        self.image_grayscale: np.ndarray = np.array([])  # Stage-1
        self.image_gaussian_blurred: np.ndarray = np.array([])  # Stage-2
        self.image_edged: np.ndarray = np.array([])  # Stage-3
        self.image_gradient_theta: np.ndarray = np.array([])  # Stage-4
        self.hstep_mat: np.ndarray = np.array([])  # Stage-5a
        self.vstep_mat: np.ndarray = np.array([])  # Stage-5b
        self.dstep_mat: np.ndarray = np.array([])  # Stage-5c
        self.image_swt: np.ndarray = np.array([])  # Stage-6
        self.letters: Dict[int, Letter] = dict()  # Reset letters dict
        self.words: Dict[int, Word] = dict()  # Reset words dict

    def _grayscaleConversion(self):
        """
        Convert the input image to gray-scale if the input_flag was `TRANSFORM_INPUT__3C_IMAGE`

        .. note::
            This is a supporting function to `transformImage`. Call to this function is made from
            `transformImage` hence using the parameters provided in `transformImage`
        """
        if self.input_flag == TRANSFORM_INPUT__3C_IMAGE:
            self.image_grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def _gaussianBlurr(self):
        """
        Apply the gaussian blurr to the gray-scale input image if the parameter `gaussian_blurr` has been
        set to `True`. The kernel used for Gaussian Blurring is taken from the parameter `gaussian_blurr_kernel`

        .. note::
            This is a supporting function to `transformImage`. Call to this function is made from `transformImage`,
            hence using the parameters provided in `transformImage` call
        """
        gaussian_blurr = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_GAUSSIAN_BLURR)
        gaussian_blurr_kernel = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_GAUSSIAN_BLURR_KERNEL)
        if gaussian_blurr:
            self.image_gaussian_blurred = cv2.GaussianBlur(self.image_grayscale, gaussian_blurr_kernel, 0)
        else:
            self.image_gaussian_blurred = np.copy(self.image_grayscale)

    def _edgeImage(self):
        """
        Apply the Edge Function to the gray-scale and gaussian blurred input image.
        If the parameter `edge_function` == 'ac', Auto Canny Edging is used, otherwise if
        an external edge function is provided to `edge_function` then that function is used
        for finding the edge of the gaussian blurred image.

        .. note::
            This is a supporting function to `transformImage`. Call to this function is made from `transformImage`,
            hence using the parameters provided in `transformImage` call
        """
        edge_function = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_EDGE_FUNCTION)
        auto_canny_sigma = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_AUTO_CANNY_SIGMA)
        if edge_function == 'ac':
            self.image_edged = auto_canny(img=self.image_gaussian_blurred, sigma=auto_canny_sigma)
        else:
            self.image_edged = edge_function(self.image_gaussian_blurred)

        self.image_edged = (self.image_edged != 0).astype(int)

    def _gradientImage(self):
        """
        This function calculates the image gradient theta angle for each pixel.

        .. note::
            This is a supporting function to `transformImage`. Call to this function is made from `transformImage`,
            hence using the parameters provided in `transformImage` call
        """
        dx = cv2.Sobel(self.image_grayscale, cv2.CV_32F, 1, 0, ksize=5, scale=-1,
                       delta=1, borderType=cv2.BORDER_DEFAULT)
        dy = cv2.Sobel(self.image_grayscale, cv2.CV_32F, 0, 1, ksize=5, scale=-1,
                       delta=1, borderType=cv2.BORDER_DEFAULT)
        self.image_gradient_theta = np.arctan2(dy, dx)
        self.image_gradient_theta = self.image_gradient_theta * self.image_edged

    def _calcStepMatrices(self):
        """
        This function calculates the step matrices for in three directions
        hstep_mat (np.ndarray) : For each pixel, cos(gradient_theta), where gradient_theta is the gradient
         angle for that pixel, representing length of horizontal movement for every unit movement in gradients direction.
         Same size as the original image
        vstep_mat (np.ndarray) : For each pixel, sin(gradient_theta), where gradient_theta is the gradient
         angle for that pixel, representing length of vertical movement for every unit movement in gradients direction.
         Same size as the original image
        dstep_mat (np.ndarray) : np.sqrt(hstep_mat**2+vstep_mat**2)

        This function also reverses the step directions for Horizontal and Vertical directions
        if the `text_mode` parameter provided is `db_lf` (Dark Background - Light Foreground).

        .. note::
            This is a supporting function to `transformImage`. Call to this function is made from `transformImage`,
            hence using the parameters provided in `transformImage` call
        """
        text_mode = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_TEXT_MODE)
        self.hstep_mat = np.round(np.cos(self.image_gradient_theta), 5)
        self.vstep_mat = np.round(np.sin(self.image_gradient_theta), 5)
        self.dstep_mat = np.round(np.sqrt(self.hstep_mat ** 2 + self.vstep_mat ** 2), 5)
        if text_mode == 'db_lf':
            self.hstep_mat *= -1
            self.vstep_mat *= -1

    def transformImage(self, text_mode: Optional[str] = 'lb_df',
                       engine: Optional[str] = 'numba',
                       gaussian_blurr: Optional[bool] = True,
                       gaussian_blurr_kernel: Optional[Tuple] = (5, 5),
                       edge_function: Optional[Union[str, Callable]] = 'ac',
                       auto_canny_sigma: Optional[float] = 0.33,
                       minimum_stroke_width: Optional[int] = 3,
                       maximum_stroke_width: Optional[int] = 200,
                       check_angle_deviation: Optional[bool] = True,
                       maximum_angle_deviation: Optional[float] = np.pi / 6,
                       include_edges_in_swt: Optional[bool] = True,
                       display: Optional[bool] = True) -> np.ndarray:
        """
        Transform the input image into its Stroke Width Transform. The entire transformation follows
        the following flow

            - Step-1 : Convert To Gray-Scale
            - Step-2 : Apply Gaussian Blurr
            - Step-3 : Find Edge of the Image
            - Step-4 : Calculate the Image Gradient Theta Angle
            - Step-5 : Calculate the Step Matrices
            - Step-6 : Apply Stroke Width Transformation

        This function also stores the time taken to complete all the above mentioned stages in the class
        attribute `transform_time`

        Args:
            text_mode (Optional[str]) : Contrast of the text present in the image, which needs to be
             transformed. Two possible values :
                1) "db_lf" :> Dark Background Light Foreground i.e Light color text on Dark color background
                2) "lb_df" :> Light Background Dark Foreground i.e Dark color text on Light color background
             This parameters affect how the gradient vectors (the direction) are calculated, since gradient
             vectors of db_lf are in  −𝑣𝑒  direction to that of lb_df gradient vectors. [default = 'lb_df']

            engine (Optional[str]) : Which engine to use for applying the Stroke Width Transform. [default = 'numba']
                1) "python" : Use `Python` for running the `findStrokes` function
                2) "numba" : Use `numba` for running the `findStrokes` function

            gaussian_blurr (Optional[bool]) : Whether to apply gaussian blurr or not. [default = True]

            gaussian_blurr_kernel (Optional[Tuple]) : Kernel to use for gaussian blurr. [default = (5, 5)]

            edge_function (Optional[str, Callable]) : Finding the Edge of the image is a tricky part, this is
             pertaining to the fact that in most of the cases the images we deal with are not of standard that
             applying just a opencv Canny operator would result in the desired Edge Image.
             Sometimes (In most cases) there is some custom processing required before edging,
             for that reason alone this parameter accepts one of the following two values :-

                1.) 'ac' :> Auto-Canny function, an in-built function which will
                            generate the Canny Image from the original image, internally
                            calculating the threshold parameters, although, to tune it
                            even further 'ac_sigma : float, default(0.33)' parameter is provided which
                            can take any value between 0.0 <--> 1.0.

                2.) A custom function : This function should have its signature as mentioned below :

                        >>> def custom_edge_func(gray_image):
                        >>>     # Your Function Logic...
                        >>>     edge_image =
                        >>>     return edge_image

            auto_canny_sigma: (Optional[float]) : Value of the sigma to be used in the edging function, if `edge_function`
             parameter is given the value "ac". [default = 0.33]

            minimum_stroke_width (Optional[int]) : Maximum permissible stroke width. [default = 0.33]

            maximum_stroke_width (Optional[int]) : Minimum permissible stroke width. [default = 0.33]

            check_angle_deviation (Optional[bool]) : Whether to check the angle deviation to terminate the ray. [default = 0.33]

            maximum_angle_deviation (Optional[float]) : Maximum Angle Deviation which would be permissible. [default = 0.33]

            include_edges_in_swt (Optional[bool]) : Whether to include edges (those edges, from which no stroke
             was able to be determined) in the final swt transform

            display (Optional[bool]) : If set to True, the images corresponding to following image codes will be displayed . [default = True]

                IMAGE_ORIGINAL = b'01' -> Original Image
                IMAGE_GRAYSCALE = b'02' -> Gray Sclaed Image
                IMAGE_EDGED = b'03' -> Edged Image
                IMAGE_SWT_TRANSFORMED = b'04' -> SWT Transformed image converted to three channels
                .. note :
                    IMAGE_SWT_TRANSFORMED is not the same as the image array returned from this function.
        Returns:
            (np.ndarray) : Stroke Width Transform of the image.
        Raise:
            SWTValueError, SWTTypeError
        Example:
        ::
            >>> # Transform the image using the default engine [default : engine='numba']
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=True)
            >>> # (A plot will be displayed as well)
            >>> print('Time Taken', swtImgObj.transform_time)
            Time Taken 0.193 sec

            >>> # Python engine been used for `transformImage` for engine = 'python'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, engine='python')
            >>> # (A plot will be displayed as well)
            >>> print('Time Taken', swtImgObj.transform_time)
            Time Taken 3.822 sec

            >>> # Wrong Input given -> SWTValueError/SWTTypeError will be raised
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='asc', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5.1, maximum_stroke_width=50)
            SWTTypeError: `minimum_stroke_width` value should be one of these types : [<class 'int'>]. Not mixed either.

            >>># Custom edge function been given to the `transformImage`
            >>> def custom_edge_func(gray_image):
            >>>     gauss_image = cv2.GaussianBlur(gray_image, (5,5), 1)
            >>>     laplacian_conv = cv2.Laplacian(gauss_image, -1, (5,5))
            >>>     canny_edge = cv2.Canny(laplacian_conv, 20, 90)
            >>>     return canny_edge
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function=custom_edge_func, gaussian_blurr_kernel=(3,3),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50)
        """
        # Reset Parameters
        self._resetSWTTransformParams()
        # Update configs
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_TEXT_MODE] = text_mode
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_ENGINE] = engine
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_GAUSSIAN_BLURR] = gaussian_blurr
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_GAUSSIAN_BLURR_KERNEL] = gaussian_blurr_kernel
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_EDGE_FUNCTION] = edge_function
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_AUTO_CANNY_SIGMA] = auto_canny_sigma
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_MINIMUM_STROKE_WIDTH] = minimum_stroke_width
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_MAXIMUM_STROKE_WIDTH] = maximum_stroke_width
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_CHECK_ANGLE_DEVIATION] = check_angle_deviation
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_MAXIMUM_ANGLE_DEVIATION] = maximum_angle_deviation
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_INCLUDE_EDGES_IN_SWT] = include_edges_in_swt
        self.cfg[CONFIG__SWTIMAGE__TRANSFORM_DISPLAY] = display

        transform_function = swt_strokes_jitted if engine == 'numba' else swt_strokes
        # Perform Type Sanity Checks
        self._transformImageSanityCheck()
        ts = time.perf_counter()
        # Convert the image to gray-scale if 3 channel image has been received.
        self._grayscaleConversion()
        # Apply gaussian blurr to the gray-scale image.
        self._gaussianBlurr()
        # Find the edge of the gray-scale image.
        self._edgeImage()
        # Calculate the gradient of the gray-scale image - first derivative @ Sobel.
        self._gradientImage()
        # Calculate the step matrices - for horizontal, vertical and diagonal.
        self._calcStepMatrices()
        # Perform transform
        self.image_swt = transform_function(edged_image=self.image_edged,
                                            hstep_mat=self.hstep_mat,
                                            vstep_mat=self.vstep_mat,
                                            dstep_mat=self.dstep_mat,
                                            max_stroke_width=maximum_stroke_width,
                                            min_stroke_width=minimum_stroke_width,
                                            image_height=self.image_height,
                                            image_width=self.image_width,
                                            check_angle_deviation=check_angle_deviation,
                                            image_gradient_theta=self.image_gradient_theta,
                                            max_angle_deviation=maximum_angle_deviation,
                                            include_edges_in_swt=include_edges_in_swt)
        self.transform_time = str(round(time.perf_counter() - ts, 3)) + ' sec'
        self.transform_stage_done = True
        if display:
            self.showImage(image_codes=[IMAGE_ORIGINAL, IMAGE_GRAYSCALE, IMAGE_EDGED, IMAGE_SWT_TRANSFORMED],
                           plot_title='SWT', plot_sup_title=f'\nTransform Time - {self.transform_time}')
        return self.image_swt.copy()

    def _transformImageSanityCheck(self):
        """
        Perform Sanity Checks for `transformImage` parameters
        Raise:
            SWTValueError, SWTTypeError
        """
        # Type Sanity checks
        perform_type_sanity_checks(cfg=self.cfg, cfg_of=CONFIG__SWTIMAGE__TRANSFORM)
        # gaussian_blurr_kernel
        gaussian_blurr_kernel = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_GAUSSIAN_BLURR_KERNEL)
        gs_blurr_kernel_length_check = len(gaussian_blurr_kernel) == 2
        gs_blurr_kernel_int_check = all([isinstance(k, int) for k in gaussian_blurr_kernel])
        gs_blurr_kernel_same_int_check = gaussian_blurr_kernel[0] == gaussian_blurr_kernel[1]
        gs_blurr_kernel_odd_gt3_check = all([k % 2 != 0 and k >= 3 for k in gaussian_blurr_kernel])
        if not (gs_blurr_kernel_length_check and gs_blurr_kernel_int_check
                and gs_blurr_kernel_same_int_check and gs_blurr_kernel_odd_gt3_check):
            raise SWTValueError(
                "`gaussian_blurr_kernel` should have same odd integers greater than 3,ex- (5,5) or (7,7) .")
        # edge_function
        edge_function = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_EDGE_FUNCTION)
        if not (edge_function == 'ac' or callable(edge_function)):
            raise SWTValueError("`edge_function` can only take `ac` or a callable function.")
        # auto_canny_sigma
        ac_sigma = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_AUTO_CANNY_SIGMA)
        if not (0.0 <= ac_sigma <= 1.0):
            raise SWTValueError("`auto_canny_sigma` can only take float values between 0.0 and 1.0")
        # maximum_angle_deviation
        if not (-np.pi / 2 <= self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_MAXIMUM_ANGLE_DEVIATION) <= np.pi / 2):
            raise SWTValueError("`maximum_angle_deviation` should be float between -90° <-> 90° (in radians)")
        # Pair Parameters
        min_sw = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_MINIMUM_STROKE_WIDTH)
        max_sw = self.cfg.get(CONFIG__SWTIMAGE__TRANSFORM_MAXIMUM_STROKE_WIDTH)
        if not (0 < min_sw < max_sw):
            raise SWTValueError(f"Condition must be satisfied : 0 < minimum_stroke_width < maximum_stroke_width")

    # ######################################### #
    #            LOCALIZE LETTERS               #
    # ######################################### #

    def localizeLetters(self, minimum_pixels_per_cc: Optional[int] = 50,
                        maximum_pixels_per_cc: Optional[int] = 10_000,
                        acceptable_aspect_ratio: Optional[float] = 0.2,
                        localize_by: Optional[str] = 'min_bbox',
                        padding_pct: Optional[float] = 0.01,
                        display: Optional[bool] = True) -> Dict[int, Letter]:
        """
        .. note::
            This function need to be run only after `SWTImage.transformImage` has been run.

        After having found and pruned the individual connected components, this function add boundaries
        to the `Letter`'s so found in the `SWTImage.transformImage`.

        Args:
            minimum_pixels_per_cc (Optional[int]) : Minimum pixels for each components to make it eligible
             for being a `Letter`. [default = 50]

            maximum_pixels_per_cc (Optional[int]) : Maximum pixels for each components to make it eligible
             for being a `Letter`. [default = 10_000]

            acceptable_aspect_ratio (Optional[float]) : Acceptable Aspect Ratio of each component to make it
             eligible for being a `Letter`. [default = 0.2]

            localize_by (Optional[str]) : Which method to localize the letters from : [default = 'min_bbox']
                1) `min_bbox` - Minimum Bounding Box (Rotating Bounding Box)
                2) `ext_bbox` - External Bounding Box
                3) `outline` - Contour

            padding_pct (Optional[float]) : How much padding to apply to each localizations [default = 0.01]

            display (Optional[bool]) : If set to True, this will display the following [default = True]
                IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS = b'11' -> Localization on Pruned RGB channel image
                IMAGE_ORIGINAL_LETTER_LOCALIZATIONS = b'12' -> Localization on Original image
                IMAGE_ORIGINAL_MASKED_LETTER_LOCALIZATIONS = b'13' -> Localization masked on original image
        Returns:
            Dict[int, Letter] : A dictionary with keys as letter labels and values as ``Letter`` class objects
        Raises:
            SWTImageProcessError, SWTValueError, SWTTypeError
        Example:
        ::
            >>> # Localizing Letters
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> # (A plot will be displayed as well)
            >>>  localized_letters = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                                maximum_pixels_per_cc=5200,
            >>>                                                localize_by='min_bbox')

            >>> # Running `localizeLetters` before having run `transformImage` -> Raises SWTImageProcessError
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> localized_letters = swtImgObj.localizeLetters(localize_by='min_bbox')
            SWTImageProcessError: `SWTImage.transformImage` must be called before this function
        """

        # Check if the transformImage stage has been done or not
        if not self.transform_stage_done:
            raise SWTImageProcessError("`SWTImage.transformImage` must be called before this function")

        # Old parameters
        check1 = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZELETTERS_MINIMUM_PIXELS_PER_CC) == minimum_pixels_per_cc
        check2 = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZELETTERS_MAXIMUM_PIXELS_PER_CC) == maximum_pixels_per_cc
        check3 = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZELETTERS_ACCEPTABLE_ASPECT_RATIO) == acceptable_aspect_ratio

        # Update configs
        self.cfg[CONFIG__SWTIMAGE__LOCALIZELETTERS_MINIMUM_PIXELS_PER_CC] = minimum_pixels_per_cc
        self.cfg[CONFIG__SWTIMAGE__LOCALIZELETTERS_MAXIMUM_PIXELS_PER_CC] = maximum_pixels_per_cc
        self.cfg[CONFIG__SWTIMAGE__LOCALIZELETTERS_ACCEPTABLE_ASPECT_RATIO] = acceptable_aspect_ratio
        self.cfg[CONFIG__SWTIMAGE__LOCALIZELETTERS_LOCALIZE_BY] = localize_by
        self.cfg[CONFIG__SWTIMAGE__LOCALIZELETTERS_PADDING_PCT] = padding_pct
        self.cfg[CONFIG__SWTIMAGE__LOCALIZELETTERS_DISPLAY] = display
        # Perform Sanity Checks
        self._localizeLettersSanityChecks()

        # Perform pruning only when the localization parameters have been changed
        # or its the first run
        if (not (check1 and check2 and check3)) or (self.pruned_num_cc == -1):
            # Reset Parameters only when the localization parameters have changed
            self._resetLocalizeLettersParams()
            self.letters: Dict[int, Letter] = dict()  # Reset letters dict
            self.words: Dict[int, Word] = dict()  # Reset words dict
            # Get unpruned properties
            _res = get_connected_components_with_stats(img=self.image_swt)
            self.unpruned_num_cc, self.unpruned_image_cc_1C, self.unpruned_cc_stats, self.unpruned_cc_centroids = _res
            # Pruning
            connected_components_labels = np.arange(self.unpruned_num_cc)
            # Pruning based on min and max number of pixels in a CC
            pixel_check = np.logical_and(self.unpruned_cc_stats[:, -1] > minimum_pixels_per_cc,
                                         self.unpruned_cc_stats[:, -1] < maximum_pixels_per_cc)
            # Pruning based on Aspect Ratio
            aspect_ratios = self.unpruned_cc_stats[:, 2] / self.unpruned_cc_stats[:, 3]
            # NOTE : Since its assumed that for letters, aspect ratio (width/height) will, almost always
            #        be < 1, i.e height of a letter will be more than the width it occupies. Therefore
            #        for all those letter where `aspect_ratios` calculated was > 1, then it will be assumed that
            #        width needs to be interchanged with height.
            aspect_ratios[aspect_ratios > 1] = 1 / aspect_ratios[aspect_ratios > 1]
            aspect_ratio_check = np.logical_and(aspect_ratios > acceptable_aspect_ratio,
                                                aspect_ratios < (1 / acceptable_aspect_ratio))

            pruning_checks = np.logical_and(aspect_ratio_check, pixel_check)
            pruned_connected_component_labels = connected_components_labels[pruning_checks]
            labels_to_be_pruned = np.setdiff1d(connected_components_labels, pruned_connected_component_labels)

            temp = self.unpruned_image_cc_1C.copy()
            to_be_pruned_mask = np.isin(temp, [k for k in labels_to_be_pruned if k != 0])
            temp[temp > 0] = 255
            rmask = gmask = bmask = temp.copy()
            self.image_cc_3C_to_be_pruned = np.dstack((rmask, gmask, bmask)).astype(np.uint8)
            self.image_cc_3C_to_be_pruned[to_be_pruned_mask, 0] = 67
            self.image_cc_3C_to_be_pruned[to_be_pruned_mask, 1] = 78
            self.image_cc_3C_to_be_pruned[to_be_pruned_mask, 2] = 232

            self.pruned_image_cc_1C = self.unpruned_image_cc_1C.copy()
            self.pruned_image_cc_1C[np.isin(self.pruned_image_cc_1C, labels_to_be_pruned)] = 0

            # Get unpruned properties
            _res = get_connected_components_with_stats(img=self.pruned_image_cc_1C)
            self.pruned_num_cc, self.pruned_image_cc_1C, self.pruned_cc_stats, self.pruned_cc_centroids = _res

        # Make the Letter objects
        orig_img = self.image.copy()
        swt_mat = self.image_swt.copy()
        pruned_cc = self.pruned_image_cc_1C.copy()

        for letter_label in np.arange(1, self.pruned_num_cc):
            letter = Letter(label=letter_label, image_height=self.image_height, image_width=self.image_width)
            letter_mask = pruned_cc == letter_label
            letter_mask = np.uint8(letter_mask)
            _ciy, _cix = letter_mask.nonzero()
            # Properties related to original image for this particular connected component
            _letter_color_values = orig_img[_ciy, _cix].copy()
            _mean_color = _letter_color_values.mean(axis=0).round(2)
            _median_color = np.median(_letter_color_values, axis=0).round(2)
            # Properties related to stroke widths seen in this component
            _component_sw_values = swt_mat[_ciy, _cix].copy()
            _sw_mean = np.mean(_component_sw_values)
            _sw_median = np.median(_component_sw_values)
            _sw_variance = np.var(_component_sw_values)
            _sw_count_dict = unique_value_counts(_component_sw_values)
            # Number of pixels this connected component occupies
            _area = self.pruned_cc_stats[letter_label, -1]
            # Contour of this connected component
            _contour = cv2.findContours(letter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _contour = _contour[0] if len(_contour) == 2 else _contour[1]
            letter._setLetterProps(area=_area, sw_mean=_sw_mean, sw_median=_sw_median, sw_var=_sw_variance,
                                   sw_counts=_sw_count_dict, color_mean=_mean_color, color_median=_median_color,
                                   outline=_contour)
            self.letters[letter_label] = letter

            self.letter_outline_done = True

        if localize_by == 'min_bbox':
            for letter_label, letter in self.letters.items():
                letter_contour = letter.outline
                rot_bbox = cv2.minAreaRect(letter_contour[0])
                _minimum_bbox_angle = rot_bbox[-1]
                _minimum_bbox_cx, _minimum_bbox_cy = np.round(rot_bbox[0], 2)
                _minimum_bbox = cv2.boxPoints(rot_bbox)

                _tr, _br, _bl, _tl = _minimum_bbox.copy()
                _d1_vec = _tr - _bl
                _d2_vec = _tl - _br
                _padding = padding_pct * np.linalg.norm(_d1_vec)
                _d1_ang = -math.atan2(_d1_vec[1], _d1_vec[0])
                _d2_ang = -math.atan2(_d2_vec[1], _d2_vec[0])

                _tr = _tr + _padding * np.array([np.cos(_d1_ang), -np.sin(_d1_ang)])
                _br = _br - _padding * np.array([-np.cos(np.pi - _d2_ang), -np.sin(np.pi - _d2_ang)])
                _bl = _bl - _padding * np.array([-np.cos(np.pi - _d1_ang), -np.sin(np.pi - _d1_ang)])
                _tl = _tl + _padding * np.array([np.cos(_d2_ang), -np.sin(_d2_ang)])
                _minimum_bbox = np.c_[_tr, _br, _bl, _tl].T.astype(int)

                # Find the point with the least x coordinate = anchor point
                _anchor_point = _minimum_bbox[np.argmax((_minimum_bbox == _minimum_bbox[:, 0].min()).sum(axis=1))]

                _minimum_bbox_height = abs(max(_minimum_bbox[:, 1]) - min(_minimum_bbox[:, 1]))
                _minimum_bbox_width = abs(max(_minimum_bbox[:, 0]) - min(_minimum_bbox[:, 0]))
                _minimum_bbox_aspect_ratio = _minimum_bbox_width / _minimum_bbox_height

                letter._setMinimumBBoxProps(min_height=_minimum_bbox_height,
                                            min_width=_minimum_bbox_width,
                                            min_cx=_minimum_bbox_cx,
                                            min_cy=_minimum_bbox_cy,
                                            min_ar=_minimum_bbox_aspect_ratio,
                                            angle=_minimum_bbox_aspect_ratio,
                                            anchor=_anchor_point,
                                            min_bbox=_minimum_bbox)
            self.letter_min_done = True
        elif localize_by == 'ext_bbox':
            pruned_cc = self.pruned_image_cc_1C.copy()
            for letter_label, letter in self.letters.items():
                letter_mask = np.uint8(pruned_cc.copy() == letter_label)
                if np.sum(letter_mask) > 0:
                    _iy, _ix = letter_mask.nonzero()
                    _max_x = max(_ix) * (1 + padding_pct)
                    _min_x = min(_ix) * (1 - padding_pct)
                    _max_y = max(_iy) * (1 + padding_pct)
                    _min_y = min(_iy) * (1 - padding_pct)
                    _tr = [_max_x, _min_y]
                    _br = [_max_x, _max_y]
                    _bl = [_min_x, _max_y]
                    _tl = [_min_x, _min_y]

                    _extreme_bbox_height = (_max_y - _min_y).round(2)
                    _extreme_bbox_width = (_max_x - _min_x).round(2)
                    _extreme_bbox_cx = _tr[0] + _extreme_bbox_width / 2
                    _extreme_bbox_cy = _tr[1] + _extreme_bbox_height / 2
                    _extreme_bbox_ar = _extreme_bbox_width / _extreme_bbox_height
                    _extreme_bbox_anchor_point = _tr
                    _extreme_bbox = np.c_[_tr, _br, _bl, _tl].T.astype(int)

                    letter._setExternalBBoxProps(ext_height=_extreme_bbox_height,
                                                 ext_width=_extreme_bbox_width,
                                                 ext_cx=_extreme_bbox_cx,
                                                 ext_cy=_extreme_bbox_cy,
                                                 ext_ar=_extreme_bbox_ar,
                                                 ext_anchor=_extreme_bbox_anchor_point,
                                                 ext_bbox=_extreme_bbox)
            self.letter_ext_done = True

        self.image_pruned_3C_letter_localized = image_1C_to_3C(self.pruned_image_cc_1C.copy())
        self.image_original_letter_localized = self.image.copy()
        self.image_original_masked_letter_localized = np.full(shape=self.image.shape, fill_value=0, dtype=np.uint8)
        for letter_label, letter in self.letters.items():
            # Add the localization for the first display - pruned_cc_1c
            self.image_pruned_3C_letter_localized = letter.addLocalization(
                image=self.image_pruned_3C_letter_localized, localize_type=localize_by,
                fill=False)
            # Add the localization for the second display - orig_img_annotation
            self.image_original_letter_localized = letter.addLocalization(
                image=self.image_original_letter_localized, localize_type=localize_by,
                fill=False)
            # Prepare the mask for the third display - orig_img_mask
            self.image_original_masked_letter_localized = letter.addLocalization(
                image=self.image_original_masked_letter_localized, localize_type=localize_by,
                fill=True)
        self.image_original_masked_letter_localized = self.image_original_masked_letter_localized / 255
        self.image_original_masked_letter_localized = self.image_original_masked_letter_localized.astype(np.uint8)
        self.image_original_masked_letter_localized = self.image_original_masked_letter_localized * self.image.copy()
        self.image_original_masked_letter_localized[self.image_original_masked_letter_localized == 0] = 255
        self.letter_stage_done = True
        if display:
            _plt_sup_title = _LETTER_SUP_TITLE_MAPPINGS.get(localize_by)
            self.showImage(image_codes=[IMAGE_CONNECTED_COMPONENTS_3C,
                                        IMAGE_CONNECTED_COMPONENTS_3C_WITH_PRUNED_ELEMENTS,
                                        IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS,
                                        IMAGE_ORIGINAL_MASKED_LETTER_LOCALIZATIONS],
                           plot_title='Letter Localizations\n',
                           plot_sup_title=rf'Localization Method : ${_plt_sup_title}$')

        return self.letters

    def _localizeLettersSanityChecks(self):
        """
        Perform Sanity Checks for `localizeLetter` parameters

        Raise:
            SWTValueError, SWTTypeError, SWTImageProcessError
        """
        perform_type_sanity_checks(cfg=self.cfg, cfg_of=CONFIG__SWTIMAGE__LOCALIZELETTERS)
        padding_pct = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZELETTERS_PADDING_PCT)
        if not (0 <= padding_pct <= 1.0):
            raise SWTValueError("`padding_pct` can take only values in the range of [0.0, 1.0]")
        min_pixels_per_cc = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZELETTERS_MINIMUM_PIXELS_PER_CC)
        max_pixels_per_cc = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZELETTERS_MAXIMUM_PIXELS_PER_CC)
        if not (0 < min_pixels_per_cc < max_pixels_per_cc):
            raise SWTValueError(f"Condition must be satisfied : 0 < minimum_pixels_per_cc < maximum_pixels_per_cc")

    def getLetter(self, key: int, localize_by: Optional[str] = 'min_bbox', display: Optional[bool] = True):
        """
        .. note::
            This function need to be run only after `localizeLetters` has been run.

        Get a particular letter being housed in `letters` attribute

        Args:
            key (int) : Letter key associated to `letters` attribute
            localize_by (Optional[str]) : Which localization to apply [default = 'min_bbox']
                1) `min_bbox` - Minimum Bounding Box (Rotating Bounding Box)
                2) `ext_bbox` - External Bounding Box
                3) `outline` - Contour
            display (Optional[bool]) : If set to True this will display the following images [default = True]
                IMAGE_INDIVIDUAL_LETTER_LOCALIZATION = b'17' -> Individual Letter Localized over Pruned RGB Image
                IMAGE_ORIGINAL_INDIVIDUAL_LETTER_LOCALIZATION = b'18' -> Individual Letter Localized over Original Image
        Returns:
            (Letter) : Individual ``Letter`` which was queried
            (np.ndarray) : Localization on Edge and SWT Image
            (np.ndarray) : Localization on Original Image
        Raises:
            SWTImageProcessError, SWTValueError
        Example:
        ::
            >>> # Localizing Letters
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixls_per_cc=5200,
            >>>                                              localize_by='min_bbox', display=False)

            >>> # Access all the letters which have been localized
            >>> swtImgObj.letters
            {1: Letter-1, 2: Letter-2, 3: Letter-3, 4: Letter-4 ...

            >>> # Accessing an individual letter by its key in `swtImgObj.letters` dictionary
            >>> _letter, _edgeswt_letter, _orig_image_letter = swtImgObj.getLetter(1, display=True)

            >>> # Accessing `getLetter` for a `localize_by` which hasn't been run already by the
            >>> # `localizeLetters` function will raise an error -> SWTImageProcessError will be raised
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letters = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                               maximum_pixels_per_cc=5200,
            >>>                                               localize_by='min_bbox', display=False)
            >>> # Accessing `min_bbox` wont raise any error as that has been run already by the localizeLetters function
            >>> _letter, _edgeswt_letter, _orig_image_letter = swtImgObj.getLetter(1, localize_by='min_bbox', display=True)
            >>> # Accessing `ext_bbox` when `ext_bbox` hasn't been run already by the localizeLetters function
            >>> _letter, _edgeswt_letter, _orig_image_letter = swtImgObj.getLetter(1, localize_by='ext_bbox', display=True)
            SWTImageProcessError: 'SWTImage.localizeLetters' with localize_by='ext_bbox' should be run before this.
            >>> # Solution : Run the `localizeLetters` function with `ext_bbox` and then access getLetter for `ext_bbox`
            >>> localized_letters = swtImgObj.localizeLetters(localize_by='ext_bbox', display=False)
            >>> _letter, _edgeswt_letter, _orig_image_letter = swtImgObj.getLetter(1, localize_by='min_bbox', display=True)
            >>> _letter, _edgeswt_letter, _orig_image_letter = swtImgObj.getLetter(1, localize_by='ext_bbox', display=True)
        """
        # Sanity Checks
        self.cfg[CONFIG__SWTIMAGE__GETLETTER_KEY] = key
        self.cfg[CONFIG__SWTIMAGE__GETLETTER_LOCALIZE_BY] = localize_by
        self.cfg[CONFIG__SWTIMAGE__GETLETTER_DISPLAY] = display
        perform_type_sanity_checks(cfg=self.cfg, cfg_of=CONFIG__SWTIMAGE__GETLETTER)
        if not self.letters:
            raise SWTImageProcessError(
                f"'SWTImage.localizeLetters' with localize_by='{localize_by}' should be run before this.")
        if key not in self.letters:
            raise SWTValueError("Invalid Key")

        edge_img = self.image_edged.copy()
        orig_img = self.image.copy()
        letter = self.letters.get(key)
        letter._checkAvailability(localize_by=localize_by)

        pruned_cc_3c = image_1C_to_3C(self.pruned_image_cc_1C)
        edge_iy, edge_ix = np.where(edge_img != 0)
        cc_iy, cc_ix = np.where(self.pruned_image_cc_1C != letter.label)
        letter_cc_3c = pruned_cc_3c.copy()
        letter_cc_3c[cc_iy, cc_ix, :] = 0  # Nullify other connected components
        letter_cc_3c[edge_iy, edge_ix, :] += 255  # Add the edged image
        self.individual_letter_localized_edgeswt = letter.addLocalization(image=letter_cc_3c,
                                                                          localize_type=localize_by, fill=False)
        self.individual_letter_localized_original = letter.addLocalization(image=orig_img,
                                                                           localize_type=localize_by, fill=False)

        if display:
            _plt_sup_title = _LETTER_SUP_TITLE_MAPPINGS.get(localize_by)
            self.showImage(image_codes=[IMAGE_INDIVIDUAL_LETTER_LOCALIZATION,
                                        IMAGE_ORIGINAL_INDIVIDUAL_LETTER_LOCALIZATION],
                           plot_title=f'Letter - {letter.label}\n',
                           plot_sup_title=rf'Localization Method : ${_plt_sup_title}$')

        return letter, self.individual_letter_localized_edgeswt, self.individual_letter_localized_original

    def letterIterator(self, localize_by: Optional[str] = 'min_bbox',
                       display: Optional[bool] = True):
        """
        .. note::
            This function can run only after `localizeLetters` has been for the particular `localize_type`.

        Generator to Iterate over all the letters in IPython/Jupyter interactive environment.

         Args:
            localize_by (Optional[str]) : Which localization to apply [defautl = 'min_bbox']
                1) `min_bbox` - Minimum Bounding Box (Rotating Bounding Box)
                2) `ext_bbox` - External Bounding Box
                3) `outline` - Contour
            display (Optional[bool]) : If set to True this will display the following images [default = True]
                IMAGE_INDIVIDUAL_LETTER_LOCALIZATION = b'17' -> Individual Letter Localized over Pruned RGB Image
                IMAGE_ORIGINAL_INDIVIDUAL_LETTER_LOCALIZATION = b'18' -> Individual Letter Localized over Original Image
        Returns:
            (Letter) : Individual ``Letter`` which was queried
            (np.ndarray) : Localization on Edge and SWT Image
            (np.ndarray) : Localization on Original Image
        Example:
        ::
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letters = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                               maximum_pixels_per_cc=5200,
            >>>                                               localize_by='min_bbox', display=False)
            >>> # (A plot will be displayed as well at every `next` call to this generator since display=True)
            >>> # Ensure the localize_by parameter has already been run in `localizeLetters` function.
            >>> localized_letter_generator = swtImgObj.letterIterator(localize_by='min_bbox', display=False)
            >>> _letter, _edgeswt_letter, _orig_image_letter = next(localized_letter_generator)
        """
        if self.letters:
            for letter_key in self.letters:
                letter, edgeswt_loc, orig_loc = self.getLetter(key=int(letter_key), localize_by=localize_by,
                                                               display=display)
                yield letter, edgeswt_loc, orig_loc

    # ######################################### #
    #             LOCALIZE WORDS                #
    # ######################################### #

    def localizeWords(self, localize_by: Optional[str] = 'bubble',
                      lookup_radius_multiplier: Optional[float] = 1.1,
                      acceptable_stroke_width_ratio: Optional[float] = 2.0,
                      acceptable_color_deviation: Optional[List] = [13, 13, 13],
                      acceptable_height_ratio: Optional[float] = 1.5,
                      acceptable_angle_deviation: Optional[float] = 30.0,
                      polygon_dilate_iterations: Optional[int] = 5,
                      polygon_dilate_kernel: Optional[int] = (5, 5),
                      display: Optional[bool] = True) -> Dict[int, Word]:
        """
        .. note::
            This function can run only after `localizeLetters` has been for the particular `localize_type="min_bbox"`.

        Once the ``letters`` attribute has been populated with the pruned connected components,
        these components can be fused together into ``Word``'s. This fusion process is taken care of
        by the ``Fusion`` class which groups a ``Letter`` with another based on comparisons such as :
            - Ratio between two individual ``Letter``'s
            - Ratio between two individual ``Letter``'s heights
            - Difference between two individual ``Letter``'s minimum bounding box rotation angle
            - Difference between two individual ``Letter``'s color vectors
        ``Letter``'s which come under consideration of being grouped for a particular ``Letter``, will be in
        the close proximity of the ``Letter``, which is gauged by components minimum bouncing box circum circle.

        Dilation is performed before finding the localization for a word when `localize_by` parameter is "polygon",
        so as to merge the nearby bounding box.

        Args:
            localize_by (Optional[str]) : One of the three localizations can be performed : [default = 'bubble']
                - 'bubble' : Bubble Boundary
                - 'bbox' : Bounding Box
                - 'polygon' : Contour Boundary
            lookup_radius_multiplier (Optional[float]) : Circum Radius multiplier, to inflate the lookup
             range. [default = 1.1]

            acceptable_stroke_width_ratio (Optional[float]) : Acceptable stroke width ratio between two ``Letter``'s
             to make them eligible to be a part of a word. [default = 2.0]

            acceptable_color_deviation (Optional[List]) : Acceptable color deviation between two ``Letter``'s to make
             them eligible to be a part of a word.. [default = [13, 13, 13]]

            acceptable_height_ratio (Optional[float]) : Acceptable height ratio between two ``Letter``'s to make them
             eligible to be a part of a word.. [default = 1.5]

            acceptable_angle_deviation (Optional[float]) : Acceptable angle deviation between two ``Letter``'s to
             make them eligible to be a part of a word.. [default = 30.0]

            polygon_dilate_iterations (Optional[int]) : Only required when localize_by = 'polygon'. Number of
             iterations to be performed before finding contour. [default = 5]

            polygon_dilate_kernel (Optional[int]) : Only required when localize_by = 'polygon', dilation kernel. [default = (5,5)]

            display (Optional[bool]) : If set tot True, this function will display . [default = 'bubble']
                IMAGE_PRUNED_3C_WORD_LOCALIZATIONS = b'14' -> Pruned RGB Image with Word Localizations
                IMAGE_ORIGINAL_WORD_LOCALIZATIONS = b'15' -> Original Image with Word Localizations
                IMAGE_ORIGINAL_MASKED_WORD_LOCALIZATIONS = b'16' -> Original Image mask with Word Localizations

        Returns:
            Dict[int, Word] : A dictionary with keys as word labels and values as ``Word`` class objects
        Raises:
            SWTImageProcessError, SWTValueError, SWTTypeError
        Example:
        ::
            >>> # To Localize Words, after having localized Letters
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixels_per_cc=5200,
            >>>                                              localize_by='min_bbox', display=False)
            >>> # (A plot will be displayed as well)
            >>> localized_words = swtImgObj.localizeWords()

            >>> # If `localizeWords` is run before having run `localizeLetters`, it will
            >>> # raise an error -> SWTImageProcessError will be raised
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_words = swtImgObj.localizeWords()
            SWTImageProcessError: `SWTImage.localizeLetters` with localize_by='min_bbox' must be called before this function

            >>> # Before running `localizeWords` its required that `localizeLetters` has been
            >>> # run with localize_by='min_bbox' parameter. Otherwise SWTImageProcessError is raised
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixels_per_cc=5200,
            >>>                                              localize_by='ext_bbox', display=False)
            >>> localized_words = swtImgObj.localizeWords()
            SWTImageProcessError: `SWTImage.localizeLetters` with localize_by='min_bbox' must be called before this function
        """
        # TODO : Add the functionality to detect whether the changes were made to the
        # TODO : localizations parameters or just annotation parameter and accordingly make the resets.
        # Check if transform stage has been run first or not
        if not self.letter_min_done:
            raise SWTImageProcessError(
                "`SWTImage.localizeLetters` with localize_by='min_bbox' must be called before this function")
        # Update configs & Initialise
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_LOCALIZE_BY] = localize_by
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_LOOKUP_RADIUS_MULTIPLIER] = lookup_radius_multiplier
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_STROKE_WIDTH_RATIO] = acceptable_stroke_width_ratio
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_COLOR_DEVIATION] = acceptable_color_deviation
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_HEIGHT_RATIO] = acceptable_height_ratio
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_ANGLE_DEVIATION] = acceptable_angle_deviation
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_POLYGON_DILATE_ITERATIONS] = polygon_dilate_iterations
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_POLYGON_DILATE_KERNEL] = polygon_dilate_kernel
        self.cfg[CONFIG__SWTIMAGE__LOCALIZEWORDS_DISPLAY] = display
        # Sanity Checks
        self._localizeWordsSanityChecks()

        # Create ProxyLetter list
        all_letters = deepcopy(self.letters)
        proxy_letters = dict()
        for letter_label, letter in all_letters.items():
            circular_mask = np.full(shape=(self.image_height, self.image_width), fill_value=0, dtype=np.uint8)
            circular_mask = letter.addLocalization(image=circular_mask, localize_type='circular',
                                                   fill=True, radius_multiplier=lookup_radius_multiplier)
            inflated_radius = np.float64(letter.min_bbox_circum_radii * lookup_radius_multiplier)
            proxy_letter = ProxyLetter(label=np.int64(letter.label),
                                       sw_median=np.float64(letter.stroke_widths_median),
                                       color_median=np.float64(letter.color_median_mag),
                                       min_height=np.float64(letter.min_bbox_height),
                                       min_angle=np.float64(letter.min_bbox_angle),
                                       inflated_radius=inflated_radius,
                                       circular_mask=circular_mask.astype(np.uint8),
                                       min_label_mask=letter.min_label_mask.astype(np.uint8))

            proxy_letters[letter_label] = proxy_letter

        # Instantiate & Run Fusion
        fusion_obj = Fusion(letters=proxy_letters,
                            acceptable_stroke_width_ratio=acceptable_stroke_width_ratio,
                            acceptable_color_deviation=acceptable_color_deviation,
                            acceptable_height_ratio=acceptable_height_ratio,
                            acceptable_angle_deviation=acceptable_angle_deviation)
        grouped_words = fusion_obj.runGrouping()

        # Prepare Words
        if not self.words:
            for label, each_word in enumerate(grouped_words):
                word = Word(label=label + 1,
                            letters=[self.letters.get(each_letter.label) for each_letter in each_word],
                            image_height=self.image_height, image_width=self.image_width)
                self.words[label + 1] = word

        # Localise
        if localize_by == 'polygon':
            for label, each_word in self.words.items():
                polygon_mask = np.full(shape=self.image_grayscale.shape, fill_value=0, dtype=np.uint8)
                for each_letter in each_word.letters:
                    polygon_mask = each_letter.addLocalization(image=polygon_mask, localize_type='min_bbox',
                                                               fill=True)
                polygon_mask = cv2.dilate(src=polygon_mask,
                                          kernel=np.ones(shape=polygon_dilate_kernel, dtype=np.uint8),
                                          iterations=polygon_dilate_iterations)
                contours = cv2.findContours(polygon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                each_word._setPolygonProps(polygon=contours)

            self.word_polygon_done = True

        elif localize_by == 'bbox':
            for label, each_word in self.words.items():
                nletters = each_word.nletters
                letters_bboxes = np.full(shape=(nletters * 4, 2), fill_value=np.nan)
                for i in range(nletters):
                    letters_bboxes[4 * i:4 * (i + 1), :] = each_word.letters[i].min_bbox.copy()
                _max_x = max(letters_bboxes[:, 0])
                _min_x = min(letters_bboxes[:, 0])
                _max_y = max(letters_bboxes[:, 1])
                _min_y = min(letters_bboxes[:, 1])
                _tr = [_max_x, _min_y]
                _br = [_max_x, _max_y]
                _bl = [_min_x, _max_y]
                _tl = [_min_x, _min_y]
                _bbox = np.c_[_tr, _br, _bl, _tl].T.astype(np.int64)
                each_word._setBBoxProps(bbox=_bbox)

            self.word_bbox_done = True

        elif localize_by == 'bubble':
            for label, each_word in self.words.items():
                circular_mask = np.full(shape=self.image_grayscale.shape, fill_value=0, dtype=np.uint8)
                for each_letter in each_word.letters:
                    circular_mask = each_letter.addLocalization(image=circular_mask, localize_type='circular',
                                                                fill=True)
                contour = cv2.findContours(circular_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = contour[0] if len(contour) == 2 else contour[1]
                each_word._setBubbleProps(bubble=contour)

            self.word_bubble_done = True

        self.image_pruned_3C_word_localized = image_1C_to_3C(self.pruned_image_cc_1C.copy())
        self.image_original_word_localized = self.image.copy()
        self.image_original_masked_word_localized = np.full(shape=self.image.shape, fill_value=0, dtype=np.uint8)

        for label, word in self.words.items():
            # Add the localization for the first display - pruned_cc_1c
            self.image_pruned_3C_word_localized = word.addLocalization(
                image=self.image_pruned_3C_word_localized, localize_type=localize_by,
                fill=False)
            # Add the localization for the second display - orig_img_annotation
            self.image_original_word_localized = word.addLocalization(
                image=self.image_original_word_localized, localize_type=localize_by,
                fill=False)
            # Prepare the mask for the third display - orig_img_mask
            self.image_original_masked_word_localized = word.addLocalization(
                image=self.image_original_masked_word_localized, localize_type=localize_by,
                fill=True)

        self.image_original_masked_word_localized = self.image_original_masked_word_localized / 255
        self.image_original_masked_word_localized = self.image_original_masked_word_localized.astype(np.uint8)
        self.image_original_masked_word_localized = self.image_original_masked_word_localized * self.image.copy()
        self.image_original_masked_word_localized[self.image_original_masked_word_localized == 0] = 255
        self.word_stage_done = True

        # Display
        if display:
            _plt_sup_title = _WORD_SUP_TITLE_MAPPINGS.get(localize_by)
            self.showImage(image_codes=[IMAGE_PRUNED_3C_WORD_LOCALIZATIONS,
                                        IMAGE_ORIGINAL_WORD_LOCALIZATIONS,
                                        IMAGE_ORIGINAL_MASKED_WORD_LOCALIZATIONS],
                           plot_title='Word Localizations\n',
                           plot_sup_title=rf'Localization Method : ${_plt_sup_title}$')

        return self.words

    def _localizeWordsSanityChecks(self):
        """
        Perform Sanity Checks for `localizeWord` parameters
        Raise:
            SWTValueError, SWTTypeError, SWTImageProcessError
        """
        perform_type_sanity_checks(cfg=self.cfg, cfg_of=CONFIG__SWTIMAGE__LOCALIZEWORDS)
        lookup_radius_multiplier = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZEWORDS_LOOKUP_RADIUS_MULTIPLIER)
        acceptable_stroke_width_ratio = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_STROKE_WIDTH_RATIO)
        acceptable_color_deviation = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_COLOR_DEVIATION)
        acceptable_height_ratio = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_HEIGHT_RATIO)
        acceptable_angle_deviation = self.cfg.get(CONFIG__SWTIMAGE__LOCALIZEWORDS_ACCEPTABLE_ANGLE_DEVIATION)
        if not (0.8 <= lookup_radius_multiplier <= 1.8):
            raise SWTValueError("`lookup_radius_multiplier` can only take values between [0.8, 1.8]")
        if not (1.00 <= acceptable_stroke_width_ratio <= 2.5):
            raise SWTValueError("`acceptable_stroke_width_ratio` can only take values between [1.0, 2.5]")
        if not (1.00 <= acceptable_height_ratio <= 1.5):
            raise SWTValueError("`acceptable_height_ratio` can only take values between [1.0, 1.5]")
        if not (0.00 <= acceptable_angle_deviation <= 35.0):
            raise SWTValueError("`acceptable_angle_deviation` can only take values between [0.0, 35.0]")
        if not all([isinstance(k, int) and k <= 50 for k in acceptable_color_deviation]):
            raise SWTValueError("`acceptable_color_deviation` can only have integer values with each value <= 50")

    def getWord(self, key, localize_by: Optional[str] = 'bubble', display: Optional[bool] = True):
        """
        .. note::
            This function can run only after `localizeWords` has been run with parameter `localize_type` parameter.

        Get a particular word being housed in `words` attribute

        Args:
            key (int) : Word key associated to `words` attribute
            localize_by (Optional[str]) : Which localization to apply
                1) `bubble` - Bubble Boundary
                2) `bbox` - Bounding Box
                3) `polygon` - Contour Boundary
            display (Optional[bool]) : If set to True, this will show [default = True]
                IMAGE_INDIVIDUAL_WORD_LOCALIZATION = b'19' -> Individual word localized over Pruned RGB Image
                IMAGE_ORIGINAL_INDIVIDUAL_WORD_LOCALIZATION = b'20' -> Individual word localized over Original Image
        Returns:
            (Word) : Individual ``Word`` which was queried
            (np.ndarray) : Localization on Edge and SWT Image
            (np.ndarray) : Localization on Original Image
        Raises:
            SWTImageProcessError, SWTValueError, SWTTypeError
        Example:
        ::
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixels_per_cc=5200,
            >>>                                              localize_by='min_bbox', display=False)
            >>> localized_words = swtImgObj.localizeWords(display=False)
            >>> # Access all the words which have been localized
            >>> swtImgObj.words
            {0: Word-0, 1: Word-1, 2: Word-2, 3: Word-3, 4: Word-4, ...
            >>> # Accessing an individual word by its key in `swtImgObj.words` dictionary
            >>> _word, _edgeswt_word, _orig_image_word = swtImgObj.getWord(1, display=True)

            >>> # Accessing `getWord` for a `localize_by` which hasn't been run already by the
            >>> # `localizeLetters` function will raise an error -> SWTImageProcessError will be raised
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixels_per_cc=5200,
            >>>                                              localize_by='min_bbox', display=False)
            >>> localized_words = swtImgObj.localizeWords(display=False)
            >>> # Accessing an individual word by its key in `swtImgObj.words` dictionary
            >>> _word, _edgeswt_word, _orig_image_word = swtImgObj.getWord(1, localize_by='polygon', display=True)
            SWTImageProcessError: 'SWTImage.localizeWords' with localize_by='polygon' should be run before this.

            >>> # Solution:  Run the `localizeWords` function with localize_by=`polygon` and then access getWord for `polygon`
            >>> localized_words = swtImgObj.localizeWords(localize_by='polygon', display=False)
            >>> _word, _edgeswt_word, _orig_image_word = swtImgObj.getWord(4, localize_by='polygon', display=True)
        """
        if not self.words:
            raise SWTImageProcessError(
                f"'SWTImage.localizeWords' with localize_by='{localize_by}' should be run before this.")
        # Sanity Checks
        self.cfg[CONFIG__SWTIMAGE__GETWORD_KEY] = key
        self.cfg[CONFIG__SWTIMAGE__GETWORD_LOCALIZE_BY] = localize_by
        self.cfg[CONFIG__SWTIMAGE__GETWORD_DISPLAY] = display
        perform_type_sanity_checks(cfg=self.cfg, cfg_of=CONFIG__SWTIMAGE__GETWORD)
        if key not in self.words:
            raise SWTValueError("Invalid Key")
        edge_img = self.image_edged.copy()
        orig_img = self.image.copy()
        word = self.words.get(key)
        word._checkAvailability(localize_by=localize_by)

        pruned_cc_3c = image_1C_to_3C(self.pruned_image_cc_1C)
        edge_iy, edge_ix = np.where(edge_img != 0)
        cc_iy, cc_ix = np.where(~np.isin(self.pruned_image_cc_1C, word.letter_labels))
        letter_cc_3c = pruned_cc_3c.copy()
        letter_cc_3c[cc_iy, cc_ix, :] = 0  # Nullify other connected components
        letter_cc_3c[edge_iy, edge_ix, :] += 255  # Add the edged image

        self.individual_word_localized_edgeswt = word.addLocalization(image=letter_cc_3c,
                                                                      localize_type=localize_by, fill=False)
        self.individual_word_localized_original = word.addLocalization(image=orig_img,
                                                                       localize_type=localize_by, fill=False)

        if display:
            _plt_sup_title = _WORD_SUP_TITLE_MAPPINGS.get(localize_by)
            self.showImage(image_codes=[IMAGE_INDIVIDUAL_WORD_LOCALIZATION,
                                        IMAGE_ORIGINAL_INDIVIDUAL_WORD_LOCALIZATION],
                           plot_title=f'Word - {word.label}\n',
                           plot_sup_title=rf'Localization Method : ${_plt_sup_title}$')

        return word, self.individual_word_localized_edgeswt, self.individual_word_localized_original

    def wordIterator(self, localize_by: Optional[str] = 'bubble', display: Optional[bool] = True):
        """
        .. note::
            This function can run only after `localizeWords` has been run with parameter `localize_type` parameter.

        Get a particular word being housed in `words` attribute

        Args:
            localize_by (Optional[str]) : Which localization to apply
                - `bubble` - Bubble Boundary
                - `bbox` - Bounding Box
                - `polygon` - Contour Boundary

            display (Optional[bool]) : If set to True, this will show [default = True]
             IMAGE_INDIVIDUAL_WORD_LOCALIZATION = b'19' -> Individual word localized over Pruned RGB Image
             IMAGE_ORIGINAL_INDIVIDUAL_WORD_LOCALIZATION = b'20' -> Individual word localized over Original Image
        Returns:
            (Word) : Individual ``Word`` which was queried
            (np.ndarray) : Localization on Edge and SWT Image
            (np.ndarray) : Localization on Original Image
        Raises:
            SWTImageProcessError, SWTValueError, SWTTypeError
        Example:
        ::
            >>> from swtloc import SWTLocalizer
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixels_per_cc=5200,
            >>>                                              localize_by='min_bbox', display=False)
            >>> localized_words = swtImgObj.localizeWords(localize_by='polygon', display=False, polygon_dilate_iterations=3)
            >>> # Creating a generator for a specific localize_by
            >>> word_iterator = swtImgObj.wordIterator(localize_by='polygon', display=True)

            >>> _word, _edgeswt_word, _orig_image_word = next(word_iterator)
        """
        if self.words:
            for word_key in self.words:
                letter, edgeswt_loc, orig_loc = self.getWord(key=word_key, localize_by=localize_by,
                                                             display=display)
                yield letter, edgeswt_loc, orig_loc

    # ######################################### #
    #            HELPER FUNCTIONS               #
    # ######################################### #

    def _available_codes(self, image_codes: List[ByteString]):
        """
        .. note::
            To see the full list of `ImageCodes` available and their meaning , look at the `showImage` function
            documentation

        Checks if the image required to render a list of image_codes is available or not.

        Args:
            image_codes (List[ByteString]) : A list of ByteStrings (Image Codes) to check for the availability
        Returns:
            (List[ByteString]) : A list of ByteStrings (Image Codes) which are available
        Raises:
            SWTImageProcessError
        """
        img_codes = []
        for each_img_code in image_codes:
            img, err_string = self._get_image_for_code(image_code=each_img_code)
            if img.size != 0:
                img_codes.append(each_img_code)
            else:
                print_in_red(text=err_string)
        if not img_codes:
            raise SWTImageProcessError(f"None of the {image_codes} are available!")

        return img_codes

    def _get_image_for_code(self, image_code: ByteString):
        """
        .. note::
            To see the full list of `ImageCodes` available and their meaning , look at the `showImage` function
            documentation

        Checks if the image required to render a list of image_codes is available or not.

        Args:
            image_code (ByteString) : Image Code
        Returns:
            (np.ndarray) : The image corresponding to the Image Code, required to rendering it.
        """
        img: np.ndarray = np.array([])
        err_string: str = ""
        code_name = CODE_VAR_NAME_MAPPINGS.get(image_code)

        if image_code == IMAGE_ORIGINAL:
            img = self.image
            err_string = 'No original image given'
        # run with transforms
        elif image_code == IMAGE_GRAYSCALE:
            img = self.image_grayscale
            err_string = 'Call .transformImage method for this Image Code to be populated'
        elif image_code == IMAGE_EDGED:
            img = self.image_edged
            err_string = 'Call .transformImage method for this Image Code to be populated'
        elif image_code == IMAGE_SWT_TRANSFORMED:
            img = image_1C_to_3C(self.image_swt.copy(), scale_with_values=True)
            err_string = 'Call .transformImage method for this Image Code to be populated'
        # localizeLetters
        elif image_code == IMAGE_CONNECTED_COMPONENTS_1C:
            img = self.unpruned_image_cc_1C
            err_string = 'Call .localizeLetters method for this Image Code to be populated'
        elif image_code == IMAGE_CONNECTED_COMPONENTS_3C:
            img = image_1C_to_3C(self.unpruned_image_cc_1C)
            err_string = 'Call .localizeLetters method for this Image Code to be populated'
        elif image_code == IMAGE_CONNECTED_COMPONENTS_3C_WITH_PRUNED_ELEMENTS:
            img = self.image_cc_3C_to_be_pruned
            err_string = 'Call .localizeLetters method for this Image Code to be populated'
        elif image_code == IMAGE_CONNECTED_COMPONENTS_PRUNED_1C:
            img = self.pruned_image_cc_1C
            err_string = 'Call .localizeLetters method for this Image Code to be populated'
        elif image_code == IMAGE_CONNECTED_COMPONENTS_PRUNED_3C:
            img = image_1C_to_3C(self.pruned_image_cc_1C)
            err_string = 'Call .localizeLetters method for this Image Code to be populated'
        elif image_code == IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS:
            img = self.image_pruned_3C_letter_localized
            err_string = 'Call .localizeLetters method for this Image Code to be populated'
        elif image_code == IMAGE_ORIGINAL_LETTER_LOCALIZATIONS:
            img = self.image_original_letter_localized
            err_string = 'Call .localizeLetters method for this Image Code to be populated'
        elif image_code == IMAGE_ORIGINAL_MASKED_LETTER_LOCALIZATIONS:
            img = self.image_original_masked_letter_localized
            err_string = 'Call .localizeLetters method for this Image Code to be populated'
        # localizeWords
        elif image_code == IMAGE_PRUNED_3C_WORD_LOCALIZATIONS:
            img = self.image_pruned_3C_word_localized
            err_string = 'Call .localizeWords method for this Image Code to be populated'
        elif image_code == IMAGE_ORIGINAL_WORD_LOCALIZATIONS:
            img = self.image_original_word_localized
            err_string = 'Call .localizeWords method for this Image Code to be populated'
        elif image_code == IMAGE_ORIGINAL_MASKED_WORD_LOCALIZATIONS:
            img = self.image_original_masked_word_localized
            err_string = 'Call .localizeWords method for this Image Code to be populated'
        # getLetter
        elif image_code == IMAGE_INDIVIDUAL_LETTER_LOCALIZATION:
            img = self.individual_letter_localized_edgeswt
            err_string = 'Call .getLetter method for this Image Code to be populated'
        elif image_code == IMAGE_ORIGINAL_INDIVIDUAL_LETTER_LOCALIZATION:
            img = self.individual_letter_localized_original
            err_string = 'Call .getLetter method for this Image Code to be populated'
        # getWord
        elif image_code == IMAGE_INDIVIDUAL_WORD_LOCALIZATION:
            img = self.individual_word_localized_edgeswt
            err_string = 'Call .getWord method for this Image Code to be populated'
        elif image_code == IMAGE_ORIGINAL_INDIVIDUAL_WORD_LOCALIZATION:
            img = self.individual_word_localized_original
            err_string = 'Call .getWord method for this Image Code to be populated'
        else:
            img = np.array([])
            err_string = 'Invalid Image Code'

        if img.size != 0:
            err_string = ''

        return img, err_string

    def saveCrop(self, save_path: str,
                 crop_of: Optional[str] = 'words',
                 crop_key: Optional[int] = 0,
                 crop_on: Optional[ByteString] = IMAGE_ORIGINAL,
                 crop_type: Optional[str] = 'bubble',
                 padding_pct: Optional[float] = 0.05):
        """
        .. note::
            - To see the full list of `ImageCodes` (value for `crop_on`) available and their meaning , look at the
            `showImage` function documentation

            - For crop_of = 'words', ensure `localizeWords` function has been run prior to this with the same `localize_type` as `crop_type`

            - For crop_of = 'letters', ensure `localizeLetters` function has been run prior to this with the same `localize_type` as `crop_type`

        Args:
            save_path (str) : The directory to save the image at

            crop_of (Optional[str]) : Generate the crop of 'letters' or 'words'. [default = 'words']

            crop_key (Optional[int]) : Which key to query from `letters` (if crop_of='letters') or `words` (if crop_of = 'words').[default = 0]

            crop_on (Optional[ByteString]) : [default = IMAGE_ORIGINAL]

            crop_type (Optional[str]) : Which localization to crop with. [default = 'bubble']
             For crop_of = 'words', available options are :
                    - bubble
                    - bbox
                    - polygon
             For crop_of = 'letters',available options are
                    - min_bbox
                    - ext_bbox
                    - outline

            padding_pct (Optional[float]) : Padding applied to each localization [default = 0.05]

        Raises:
            SWTValueError, SWTImageProcessError, SWTTypeError
        Example:
        ::
            >>> from swtloc import SWTLocalizer
            >>> from swtloc.configs import IMAGE_PRUNED_3C_WORD_LOCALIZATIONS
            >>> from swtloc.configs import IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS
            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixels_per_cc=5200,
            >>>                                              display=False)
            >>> localized_words = swtImgObj.localizeWords(display=False)
            >>> # To generate and save the crops of `letters`
            >>> swtImgObj.saveCrop(save_path='../', crop_of='letters', crop_key=3, crop_on=IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS,
            >>>                    crop_type='outline', padding_pct=0.01)
            >>> # To generate and save the crops of `words`
            >>> swtImgObj.saveCrop(save_path='../', crop_of='words', crop_key=8, crop_on=IMAGE_PRUNED_3C_WORD_LOCALIZATIONS,
            >>>                    crop_type='bubble', padding_pct=0.01)


            >>> # An error will be raised if `.saveCrops` functions is called for `crop_of='letters'`
            >>> # even before `.localizeLetters` for localize_by = crop_type hasn't been called before
            >>> # -> SWTImageProcessError will be raised
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> swtImgObj.saveCrop(save_path='../', crop_of='letters', crop_key=3, crop_on=IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS,
            >>>                    crop_type='outline', padding_pct=0.01)
            Call .localizeLetters method for this Image Code to be populated
            SWTImageProcessError: None of the [b'11'] are available!

            >>> # An error will be raised if `.saveCrops` functions is called for `crop_of='words'`
            >>> # even before `.localizeWords` for localize_by = crop_type hasn't been called before
            >>> # -> SWTImageProcessError will be raised
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixels_per_cc=5200,
            >>>                                              display=False)
            >>> swtImgObj.saveCrop(save_path='../', crop_of='words', crop_key=8, crop_on=IMAGE_PRUNED_3C_WORD_LOCALIZATIONS,
            >>>                    crop_type='bubble', padding_pct=0.01)
            Call .localizeWords method for this Image Code to be populated
            SWTImageProcessError: None of the [b'14'] are available!
        """
        self.cfg[CONFIG__SWTIMAGE__SAVECROPS_SAVE_PATH] = save_path
        self.cfg[CONFIG__SWTIMAGE__SAVECROPS_CROP_OF] = crop_of
        self.cfg[CONFIG__SWTIMAGE__SAVECROPS_CROP_KEY] = crop_key
        self.cfg[CONFIG__SWTIMAGE__SAVECROPS_CROP_ON] = crop_on
        perform_type_sanity_checks(cfg=self.cfg, cfg_of=CONFIG__SWTIMAGE__SAVECROPS)
        # Check if save_path exists
        if not os.path.isdir(save_path):
            raise SWTValueError(f"{save_path} not a directory!")
        # Check if crop_on exists
        _ = self._available_codes(image_codes=[crop_on])
        img, err_string = self._get_image_for_code(image_code=crop_on)
        if err_string != '':
            raise SWTImageProcessError(err_string)

        img = img.copy()
        img_mask = np.full(shape=img.shape, fill_value=0, dtype=np.uint8)

        if crop_of == 'words':
            if not self.word_stage_done:
                raise SWTImageProcessError(
                    f"'SWTImage.localizeWords' with localize_by='{crop_type}' should be run before this.")
            if crop_key not in self.words:
                raise SWTValueError("Invalid `crop_key` for `words`")
            word = self.words.get(crop_key)
            # Check for crop_type
            if crop_type in ["polygon", "bbox", "bubble"]:
                word._checkAvailability(localize_by=crop_type)
            else:
                raise SWTValueError(
                    "`crop_type` can only take one of ['polygon', 'bbox', 'bubble'] for crop_of = 'words'")
            img_mask = word.addLocalization(image=img_mask, localize_type=crop_type, fill=True)
            img = word.addLocalization(image=img, localize_type=crop_type, fill=False)

        elif crop_of == 'letters':
            if not self.letter_stage_done:
                raise SWTImageProcessError(
                    f"'SWTImage.localizeLetters' with localize_by='{crop_type}' should be run before this.")
            if crop_key not in self.letters:
                raise SWTValueError("Invalid `crop_key` for `letters`")
            letter = self.letters.get(crop_key)
            if crop_type in ["outline", "ext_bbox", "min_bbox"]:
                letter._checkAvailability(localize_by=crop_type)
            else:
                raise SWTValueError(
                    "`crop_type` can only take one of ['outline', 'ext_bbox', 'min_bbox'] for crop_of = 'letters'")

            img_mask = letter.addLocalization(image=img_mask, localize_type=crop_type, fill=True)
            img = letter.addLocalization(image=img, localize_type=crop_type, fill=False)

        img[np.where(img_mask == 0)] = 0

        if len(img_mask.shape) == 2:
            _iy, _ix = img_mask.nonzero()
        else:
            _iy, _ix, _ = img_mask.nonzero()

        _max_x = int(max(_ix) * (1 + padding_pct))
        _min_x = int(min(_ix) * (1 - padding_pct))
        _max_y = int(max(_iy) * (1 + padding_pct))
        _min_y = int(min(_iy) * (1 - padding_pct))

        if len(img_mask.shape) == 2:
            crop = img[_min_y:_max_y, _min_x:_max_x]
            _ = plt.imshow(crop, cmap='gray')
        else:
            crop = img[_min_y:_max_y, _min_x:_max_x, :]
            _ = plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        plt.savefig(
            save_path + f'{self.image_name}_{crop_of}-{crop_key}_{crop_type}_{CODE_VAR_NAME_MAPPINGS.get(crop_on)}_CROP.jpg',
            dpi=200)
        plt.close()

    def showImage(self, image_codes: Optional[List[ByteString]] = None,
                  plot_title: Optional[str] = 'SWTImage Plot',
                  plot_sup_title: Optional[str] = '',
                  save_dir: Optional[str] = '',
                  save_fig: Optional[bool] = False,
                  dpi: Optional[int] = 200):
        """
        Function to display a group of ImageCodes (maximum 4), explanation for those codes can be
        found in the table below :
        .. csv-table::
            :header: Image Code, Explanation

            IMAGE_ORIGINAL,  "Original Image"
            IMAGE_GRAYSCALE, "Gray-Scaled Image"
            IMAGE_EDGED, "Edge Image"
            IMAGE_SWT_TRANSFORMED, "SWT Transformed Image"
            IMAGE_CONNECTED_COMPONENTS_1C, "Connected Components Single Channel"
            IMAGE_CONNECTED_COMPONENTS_3C, "Connected Components RGB Channel"
            IMAGE_CONNECTED_COMPONENTS_3C_WITH_PRUNED_ELEMENTS, "Connected Components Regions which were pruned (in red)"
            IMAGE_CONNECTED_COMPONENTS_PRUNED_1C, "Pruned Connected Components Single Channel"
            IMAGE_CONNECTED_COMPONENTS_PRUNED_3C, "Pruned Connected Components RGB Channel"
            IMAGE_CONNECTED_COMPONENTS_OUTLINE, "Connected Components Outline"
            IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS, "Pruned RGB Channel SWT Image With Letter Localizations"
            IMAGE_ORIGINAL_LETTER_LOCALIZATIONS, "Original Image With Letter Localizations"
            IMAGE_ORIGINAL_MASKED_LETTER_LOCALIZATIONS, "Original Image With Masked Letter Localizations"
            IMAGE_PRUNED_3C_WORD_LOCALIZATIONS, "Pruned RGB Channel SWT Image With Words Localizations"
            IMAGE_ORIGINAL_WORD_LOCALIZATIONS, "Original Image With Words Localizations"
            IMAGE_ORIGINAL_MASKED_WORD_LOCALIZATIONS, "Original Image With Masked Words Localizations"
            IMAGE_INDIVIDUAL_LETTER_LOCALIZATION, "Individual Letter With Localizations Over Edged + SWT"
            IMAGE_ORIGINAL_INDIVIDUAL_LETTER_LOCALIZATION, "Individual Letter With Localizations Over Original"
            IMAGE_INDIVIDUAL_WORD_LOCALIZATION, "Individual Word With Localizations Over Edged + SWT"
            IMAGE_ORIGINAL_INDIVIDUAL_WORD_LOCALIZATION, "Individual Word With Localizations Over Original"

        Args:
            image_codes (Optional[List[ByteString]]) : List of image codes to display. [default = IMAGE_ORIGINAL]
            plot_title (Optional[str]) : Title of the plot
            plot_sup_title (Optional[str]) : Sub title of the plot
            save_dir (Optional[str]) : Directory in which to save the prepared plot
            save_fig (Optional[bool]) : Whether to save the prepared plot or not
            dpi (Optional[int]) : DPI of the figure to be saved
        Raise:
            SWTValueError, SWTTypeError
        Returns:
            (str) : Returns the location where the image was saved if save_dir=True and save_path is given.
        Example:
        ::
            >>> from swtloc import SWTLocalizer
            >>> from swtloc.configs import IMAGE_ORIGINAL
            >>> from swtloc.configs import IMAGE_SWT_TRANSFORMED
            >>> from swtloc.configs import IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS

            >>> root_path = 'examples/images/'
            >>> swtl = SWTLocalizer(image_paths=root_path+'test_image_1/test_img1.jpg')
            >>> swtImgObj = swtl.swtimages[0]
            >>> swt_image = swtImgObj.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi/2,
            >>>                                      edge_function='ac', gaussian_blurr_kernel=(11, 11),
            >>>                                      minimum_stroke_width=5, maximum_stroke_width=50, display=False)
            >>> localized_letter = swtImgObj.localizeLetters(minimum_pixels_per_cc=950,
            >>>                                              maximum_pixels_per_cc=5200,
            >>>                                              display=False)
            >>> swtImgObj.showImage(image_codes=[IMAGE_ORIGINAL,
            >>>                                  IMAGE_SWT_TRANSFORMED,
            >>>                                  IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS],
            >>>                     plot_title="Process Flow",
            >>>                     plot_sup_title="Original -> SWT -> Pruned Letters")

            >>> # (A plot will be displayed as well) + Save the prepared plot
            >>> localized_letter = swtImgObj.localizeLetters(display=False)
            >>> swtImgObj.showImage(image_codes=[IMAGE_ORIGINAL,
            >>>                                  IMAGE_SWT_TRANSFORMED,
            >>>                                  IMAGE_PRUNED_3C_LETTER_LOCALIZATIONS],
            >>>                     plot_title="Process Flow",
            >>>                     plot_sup_title="Original -> SWT -> Pruned Letters",
            >>>                     save_dir='../', save_fig=True, dpi=130)
        """
        if not image_codes:
            image_codes = [IMAGE_ORIGINAL]

        self.cfg[CONFIG__SWTIMAGE__SHOWIMAGE_IMAGE_CODES] = image_codes
        self.cfg[CONFIG__SWTIMAGE__SHOWIMAGE_PLOT_TITLE] = plot_title
        self.cfg[CONFIG__SWTIMAGE__SHOWIMAGE_PLOT_SUP_TITLE] = plot_sup_title
        self.cfg[CONFIG__SWTIMAGE__SHOWIMAGE_SAVE_DIR] = save_dir
        self.cfg[CONFIG__SWTIMAGE__SHOWIMAGE_SAVE_FIG] = save_fig
        self.cfg[CONFIG__SWTIMAGE__SHOWIMAGE_DPI] = dpi
        perform_type_sanity_checks(cfg=self.cfg, cfg_of=CONFIG__SWTIMAGE__SHOWIMAGE)

        if save_fig:
            # Check if save_path exists
            if not os.path.isdir(save_dir):
                raise SWTValueError(f"{save_dir} not a directory!")

        individual_plot_titles = [get_code_descriptions(k) for k in image_codes]
        individual_images = []

        # Check if all the image_codes are available
        image_codes = self._available_codes(image_codes=image_codes)

        for each_img_code in image_codes:
            individual_plot_titles.append(get_code_descriptions(each_img_code))
            img, err_string = self._get_image_for_code(image_code=each_img_code)
            if err_string != '':
                raise SWTImageProcessError(err_string)
            individual_images.append(img)

        prep_image = show_N_images(images=individual_images,
                                   plot_title=plot_title,
                                   sup_title=plot_sup_title,
                                   individual_titles=individual_plot_titles,
                                   return_img=save_fig)

        if prep_image:
            _identifier = "_".join([bytes(k).decode("utf-8") for k in image_codes])
            spath = save_dir + f'{self.image_name}_{_identifier}.jpg'
            plt.savefig(spath, dpi=dpi)
            plt.close()
            return spath
