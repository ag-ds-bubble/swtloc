# Author : Achintya Gupta
# Purpose : Base Classes

import numpy as np

try:
    from cv2 import cv2
except:
    import cv2
from typing import Union
from typing import List
from typing import Dict
from typing import ByteString


class IndividualComponentBase:
    """
    Base class representing the Individual Components found in an transformed image
    for example: a *Letter*.
    """

    def __init__(self, label: int, image_height: int, image_width: int):
        """
        Create an ``IndividualComponentBase`` object which will house the components
        properties such as :
            - Minimum Bounding Box & its related properties
            - External Bounding Box & its related properties
            - Outline (Contour)
            - Original Image Properties

        Args:
            label (int) : A unique identifier for this Component
            image_height (int) : Height of the image in which this component resides
            image_width (int) : Width of the image in which this component resides
        """
        # Identifier of this individual component
        self.label: int = label
        # Contour of this individual component
        self.outline: list = []
        # Number of pixels in this individual component
        self.area_pixels: int = -1
        # Magnitude of the median color vectors `self.original_color_median`
        self.color_median_mag: float = 0.0
        # Width of the image in which this individual component resides
        self.image_width: int = image_width
        # Mean color per-channel in this individual component
        self.original_color_mean: np.ndarray = np.array([-1, -1, -1])
        # Width of the image in which this individual component resides
        self.image_height: int = image_height
        # Median color per-channel in this individual component
        self.original_color_median: np.ndarray = np.array([-1, -1, -1])

        # # Minimum Bounding Box
        # Minimum Bounding Box of this individual component
        self.min_bbox: np.ndarray = np.array([])
        # Minimum Bounding Box Centre-X of this individual component
        self.min_bbox_cx: int = -1
        # Minimum Bounding Box Centre-Y of this individual component
        self.min_bbox_cy: int = -1
        # Minimum Bounding Box Filled Mask with label of this individual component
        self.min_label_mask: np.ndarray = np.array([])
        # Minimum Bounding Box Width of this individual component
        self.min_bbox_width: int = -1
        # Minimum Bounding Box Rotation angle with vertical of this individual component
        self.min_bbox_angle: float = 0.0
        # np.array([label, centre_x, centre_y]) of this individual component
        self.label_n_centre: np.ndarray = np.array([])
        # Minimum Bounding Box Height of this individual component
        self.min_bbox_height: int = -1
        # Minimum Bounding Box Centre co-ordinates of this individual component
        self.min_bbox_centre: np.ndarray = np.array([])
        # Minimum Bounding Box Aspect Ration of this individual component
        self.min_bbox_aspect_ratio: float = 0.0
        # Minimum Bounding Box Top Left Point
        self.min_bbox_anchor_point: np.ndarray = np.array([])
        # Minimum Bounding Box Circum Radius of this individual component
        self.min_bbox_circum_radii: float = 0.0

        # # Extreme Bounding Box
        # External Bounding Box of this individual component
        self.ext_bbox: np.ndarray = np.array([])
        # External Bounding Box Centre-X of this individual component
        self.ext_bbox_cx: int = -1
        # External Bounding Box Centre-Y of this individual component
        self.ext_bbox_cy: int = -1
        # External Bounding Box Width of this individual component
        self.ext_bbox_width: int = -1
        # External Bounding Box Height of this individual component
        self.ext_bbox_height: int = -1
        # External Bounding Box Aspect Ratio of this individual component
        self.ext_bbox_aspect_ratio: float = 0.0
        # External Bounding Box Top Left Point
        self.ext_bbox_anchor_point: np.ndarray = np.array([])

    def __repr__(self):
        """Representational String"""
        return f"IndividualComponent-{self.label}"

    def _setIcProps(self, area: int, color_mean: np.ndarray, color_median: np.ndarray,
                    outline: Union[List, np.ndarray]):
        """
        Set the Individual Component generic properties, such as the area, color_mean,
        color_median and outline. These properties are calculated after masking out everything
        else but the Individual Component.

        Args:
            area (int) : Number of pixels corresponding to this individual component
            color_mean (np.ndarray) : Mean of the individual component in the original image
            color_median (np.ndarray) : Median of the individual component in the original image
            outline (Union[List, np.ndarray]) : Outline of the individual component in the original image
        """
        self.area_pixels = area
        self.original_color_mean = color_mean
        self.original_color_median = color_median
        self.color_median_mag = np.linalg.norm(color_median)
        self.outline = outline

    def _setMinimumBBoxProps(self, min_height: int, min_width: int, min_cx: int, min_cy: int,
                             min_ar: float, angle: float, anchor: np.ndarray, min_bbox: np.ndarray):
        """
        Set the Individual Component Minimum Bounding Box properties, where a minimum bounding box
        is a rotated bounding box which completely contains this individual component. All these
        initialisations are available as attributes in this class, apart from these the following
        are available as well :

        - min_bbox_circum_radii (float) : Circum Radius of the Minimum Bounding Box
        - min_bbox_centre (np.ndarray) : Centre Co-Ordinate of the Minimum Bounding Box
        - label_n_centre (np.ndarray) : np.array([label, cx, cy])
        - min_label_mask (np.ndarray) : Boolean mask of the filled Minimum Bounding Box in a numpy
        array filled with 0's

        Args:
            min_height (int) : Height of the Minimum Bounding Box (Attribute : `min_bbox_height`)
            min_width (int) : Width of the Minimum Bounding Box (Attribute : `min_bbox_width`)
            min_cx (int) : Centre-X ordinate of the Minimum Bounding Box (Attribute : `min_bbox_cx`)
            min_cy (int) : Centre-Y ordinate of the Minimum Bounding Box (Attribute : `min_bbox_cy`)
            min_ar (float) : Aspect Ratio the Minimum Bounding Box (Attribute : `min_bbox_aspect_ratio`)
            angle (float) : Angle of the Minimum Bounding Box (Attribute : `min_bbox_angle`)
            anchor (np.ndarray) : Anchor (Top Left Co-Ordinate) of the Minimum Bounding Box (Attribute : `min_bbox_anchor_point`)
            min_bbox (np.ndarray) : Minimum Bounding Box (Attribute : `min_bbox`)
        """
        self.min_bbox_height = min_height
        self.min_bbox_width = min_width
        self.min_bbox_cx = min_cx
        self.min_bbox_cy = min_cy
        self.min_bbox_aspect_ratio = min_ar
        self.min_bbox_angle = angle
        self.min_bbox_anchor_point = anchor
        self.min_bbox = min_bbox
        self.min_bbox_circum_radii = np.sqrt(min_height ** 2 + min_width ** 2) / 2
        self.min_bbox_centre = np.array([min_cx, min_cy])
        self.label_n_centre = np.array([self.label, min_cx, min_cy])
        self.min_label_mask = np.full(shape=(self.image_height, self.image_width), fill_value=0, dtype=np.uint8)
        self.min_label_mask = self.addLocalization(image=self.min_label_mask, localize_type='min_bbox', fill=True)
        self.min_label_mask = self.min_label_mask / 255
        self.min_label_mask = self.min_label_mask * self.label

    def _setExternalBBoxProps(self, ext_height: int, ext_width: int, ext_cx: int, ext_cy: int,
                              ext_ar: float, ext_anchor: list, ext_bbox: np.ndarray):
        """
        Set the Individual Component External Bounding Box properties, where a external bounding box
        is a erect bounding box which completely contains this individual component,calculated using the
        individual component extremes. All these initialisations are available as attributes in this class,
        apart from these the following are available as well :

        Args:
            ext_height (int) : Height of the External Bounding Box (Attribute : `ext_bbox_height`)
            ext_width (int) : Width of the External Bounding Box (Attribute : `ext_bbox_width`)
            ext_cx (int) : Centre-X ordinate of the External Bounding Box (Attribute : `ext_bbox_cx`)
            ext_cy (int) : Centre-Y ordinate of the External Bounding Box (Attribute : `ext_bbox_cy`)
            ext_ar (float) : Aspect Ratio the External Bounding Box (Attribute : `ext_bbox_aspect_ratio`)
            ext_anchor (np.ndarray) : Anchor (Top Left Co-Ordinate) of the External Bounding Box (Attribute : `ext_bbox_anchor_point`)
            ext_bbox (np.ndarray) : External Bounding Box (Attribute : `ext_bbox`)
        """
        self.ext_bbox_height = ext_height
        self.ext_bbox_width = ext_width
        self.ext_bbox_cx = ext_cx
        self.ext_bbox_cy = ext_cy
        self.ext_bbox_aspect_ratio = ext_ar
        self.ext_bbox_anchor_point = ext_anchor
        self.ext_bbox = ext_bbox

    def addLocalization(self, image: np.ndarray, localize_type: str,
                        fill: bool, radius_multiplier: float = 1.0) -> np.ndarray:
        """
        Add a specific `localize_type` of localization to the input `image`. `fill` parameter tells whether to
        fill the component or not.

        Args:
            image (np.ndarray) : Image on which localization needs to be added
            localize_type (str) : Type of the localization that will be added. Can be only one of
             ['min_bbox', 'ext_bbox', 'outline', 'circular']. Where :
                - `min_bbox` : Minimum Bounding Box
                - `ext_bbox` : External Bounding Box
                - `outline` : Contour
                - `circular` : Circle - With Minimum Bounding Box Centre coordinate and
                 radius = Minimum Bounding Box Circum Radius * radius_multiplier
            fill (bool) : Whether to fill the added localization or not
            radius_multiplier (float) : Minimum Bounding Box Circum Radius inflation parameter. [default = 1.0].
        Returns:
            (np.ndarray) - annotated image
        """
        _color = (0, 0, 255)
        if fill:
            _color = (255, 255, 255)
        _thickness = (np.sqrt(self.image_height ** 2 + self.image_width ** 2)) * (4 / np.sqrt(768 ** 2 + 1024 ** 2))
        _thickness = int(_thickness)
        if _thickness == 0:
            _thickness = 1
        if localize_type == 'min_bbox' and not fill:
            image = cv2.polylines(img=image, pts=[self.min_bbox], isClosed=True, color=_color, thickness=_thickness)
        elif localize_type == 'ext_bbox' and not fill:
            image = cv2.polylines(img=image, pts=[self.ext_bbox], isClosed=True, color=_color, thickness=_thickness)
        elif localize_type == 'outline' and not fill:
            image = cv2.polylines(img=image, pts=self.outline, isClosed=True, color=_color, thickness=_thickness)
        elif localize_type == 'min_bbox' and fill:
            image = cv2.fillPoly(img=image, pts=[self.min_bbox], color=_color)
        elif localize_type == 'ext_bbox' and fill:
            image = cv2.fillPoly(img=image, pts=[self.ext_bbox], color=_color)
        elif localize_type == 'outline' and fill:
            image = cv2.fillPoly(img=image, pts=self.outline, color=_color)
        elif localize_type == 'circular' and fill:
            image = cv2.circle(img=image, center=tuple(np.uint32(self.min_bbox_centre)),
                               radius=np.uint32(self.min_bbox_circum_radii * radius_multiplier), color=255,
                               thickness=-1)
        return image


class GroupedComponentsBase:
    """
    Base class representing the Grouped Components found in an transformed image
    for example: a Word.
    """

    def __init__(self, label: int, image_height: int, image_width: int):
        """
        Create an ``IndividualComponentBase`` object which will house the grouped components
        properties such as :
            - Various Bounding Shapes which house that particular grouped component entirely
        Args:
            label (int) : A unique identifier for this Component
            image_height (int) : Height of the image in which this component resides
            image_width (int) : Width of the image in which this component resides
        """
        # Initialise image properties
        self.image_height = image_height
        self.image_width = image_width
        # Identifier for this grouped component
        self.label: int = label
        # Grouped component Bounding Box
        self.bbox: np.ndarray = np.ndarray([])
        # Grouped component Contour
        self.polygon: np.ndarray = np.ndarray([])
        # Grouped component Bubble
        self.bubble: np.ndarray = np.ndarray([])

    def __repr__(self):
        """Representational String"""
        return f"GroupedComponentBase-{self.label}"

    def _setBBoxProps(self, bbox: np.ndarray):
        """
        Set the Grouped Component External Bounding Box, where a external bounding box
        is a erect bounding box which completely contains this grouped component, calculated using the
        grouped component extremes.

        Args:
            bbox (np.ndarray) : Bounding Box (Attribute : `bbox`)
        """
        self.bbox = bbox

    def _setPolygonProps(self, polygon: Union[List, np.ndarray]):
        """
        Set the Grouped Component External Polygon Bounding, where a polygon boundary
        is a convex polygon completely containing this grouped component, calculated using contour
        post mask dilation.

        Args:
            polygon (np.ndarray) : Bounding Box (Attribute : `bbox`)
        """
        self.polygon = polygon

    def _setBubbleProps(self, bubble: Union[List, np.ndarray]):
        """
        Set the Grouped Component Bubble Boundary, where a bubble boundary
        is a convex polygon made by fusing circular masks of each individual component(Components which
        belong to this Grouped Component)

        Args:
            bubble (np.ndarray) : Bubble Boundary (Attribute : `bubble`)
        """
        self.bubble = bubble

    def addLocalization(self, image: np.ndarray, localize_type: str, fill: bool) -> np.ndarray:
        """
        Add a specific `localize_type` of localization to the input `image`. `fill` parameter tells whether to
        fill the component or not.

        Args:
            image (np.ndarray) : Image on which localization needs to be added
            localize_type (str) : Type of the localization that will be added. Can be only one of
             ['bbox', 'bubble', 'polygon']. Where
                - `bbox` : Bounding Box
                - `bubble` : Bubble Boundary
                - `polygon` : Contour Boundary
            fill (bool) : Whether to fill the added localization or not
        Returns:
            (np.ndarray) - annotated image
        """
        _color = (0, 0, 255)
        if fill:
            _color = (255, 255, 255)
        _thickness = (np.sqrt(self.image_height ** 2 + self.image_width ** 2)) * (4 / np.sqrt(768 ** 2 + 1024 ** 2))
        _thickness = int(_thickness)
        if _thickness == 0:
            _thickness = 1
        if localize_type == 'bbox' and not fill:
            image = cv2.polylines(img=image, pts=[self.bbox], isClosed=True, color=_color, thickness=_thickness)
        elif localize_type == 'bubble' and not fill:
            image = cv2.polylines(img=image, pts=self.bubble, isClosed=True, color=_color, thickness=_thickness)
        elif localize_type == 'polygon' and not fill:
            image = cv2.polylines(img=image, pts=self.polygon, isClosed=True, color=_color, thickness=_thickness)
        elif localize_type == 'bbox' and fill:
            image = cv2.fillPoly(img=image, pts=[self.bbox], color=_color)
        elif localize_type == 'bubble' and fill:
            image = cv2.fillPoly(img=image, pts=self.bubble, color=_color)
        elif localize_type == 'polygon' and fill:
            image = cv2.fillPoly(img=image, pts=self.polygon, color=_color)
        return image


class TextTransformBase:
    """
    Base class for various transformation classes.
    for example: a SWTImage.
    """

    def __init__(self, image: np.ndarray, image_name: str, input_flag: ByteString, cfg: Dict):
        """
        Create a `TextTransformBase` which will house the properties of the input image, its name, its
        input type flag, transform configuration and various other parameters corresponding to various stages
        in the transformation process.

        Args:
            image (np.ndarray) : Input image on which transformation will be performed
            image_name (str) : Name of the input images (Needed while saving the post-transformation results)
            input_flag (ByteString) : Flag of input type. It can be only one of the following
                - TRANSFORM_INPUT__1C_IMAGE = b'21'
                - TRANSFORM_INPUT__3C_IMAGE = b'22'
                These image codes reside in configs.py file
            cfg (dict) : Configuration of a particular transformation type.
        """
        # Initialisations
        # Image on which transformation has to be done
        self.image: np.ndarray = image
        # Height of the input image
        self.image_height: int = self.image.shape[0]
        # Width of the input image
        self.image_width: int = self.image.shape[1]
        # Name of the input image
        self.image_name: str = image_name
        # Input image type flag
        self.input_flag: ByteString = input_flag
        # Configuration which would be managing this transformation
        self.cfg: dict = cfg

        # > Parameters for transformImage
        # Transformation time
        self.transform_time: str = ''
        # Flag to reflect if the transform stage is done
        self.transform_stage_done: bool = False

        # > Parameters for localizing letters
        # Number of Connected Components before pruning
        self.unpruned_num_cc: int = -1
        # Image of Connected Components - Single Channel
        self.unpruned_image_cc_1C: np.ndarray = np.array([])
        # Statistics of Connected Components before pruning
        self.unpruned_cc_stats: np.ndarray = np.array([])
        # Centroids of Connected Components before pruning
        self.unpruned_cc_centroids: np.ndarray = np.array([])
        # Image of Connected Components to be pruning
        self.image_cc_3C_to_be_pruned: np.ndarray = np.array([])
        # Pruned Number of Connected Components
        self.pruned_num_cc: int = -1
        # Pruned Image of Connected Components - Single Channel
        self.pruned_image_cc_1C: np.ndarray = np.array([])
        # Statistics of the Pruned Image of Connected Components
        self.pruned_cc_stats: np.ndarray = np.array([])
        # Centroids of the Pruned Image of Connected Components
        self.pruned_cc_centroids: np.ndarray = np.array([])
        # RGB Channel pruned components which qualified as letters
        self.image_pruned_3C_letter_localized: np.ndarray = np.array([])
        # Original image pruned components which qualified as letters
        self.image_original_letter_localized: np.ndarray = np.array([])
        # Original image with only masked letters
        self.image_original_masked_letter_localized: np.ndarray = np.array([])
        # Individual letter annotated on EDGE Image + SWT Image
        self.individual_letter_localized_edgeswt: np.ndarray = np.array([])
        # Individual letter annotated on Original Image
        self.individual_letter_localized_original: np.ndarray = np.array([])
        # Flags for different stages and `localize_by`
        self.letter_stage_done: bool = False
        self.letter_min_done: bool = False
        self.letter_ext_done: bool = False
        self.letter_outline_done: bool = False

        # > Parameters for localizing words
        # RGB Channel pruned components which qualified as words
        self.image_pruned_3C_word_localized: np.ndarray = np.array([])
        # Original image pruned components which qualified as words
        self.image_original_word_localized: np.ndarray = np.array([])
        # Original image with only masked words
        self.image_original_masked_word_localized: np.ndarray = np.array([])
        # Individual word annotated on EDGE Image + SWT Image
        self.individual_word_localized_edgeswt: np.ndarray = np.array([])
        # Individual word annotated on Original Image
        self.individual_word_localized_original: np.ndarray = np.array([])
        # Flags for different stages and `localize_by`
        self.word_stage_done: bool = False
        self.word_bubble_done: bool = False
        self.word_polygon_done: bool = False
        self.word_bbox_done: bool = False

    # ######################################### #
    #                  TRANSFORM                #
    # ######################################### #
    def _resetTransformParams(self):
        """
        Resets the Transform stage parameters and the downstream stage parameters :
         - findAndPrune Parameters
         - localizeLetters Parameters
         - localizeWords Parameters
        """
        # Reset downstream functions
        self._resetLocalizeLettersParams()
        self._resetLocalizeWordsParams()

        # > Parameters for transformImage
        self.transform_time: str = ''
        self.transform_stage_done: bool = False

    # ######################################### #
    #            LOCALIZE LETTERS               #
    # ######################################### #
    def _resetLocalizeLettersParams(self):
        """
        Resets the localizeLetters stage parameters and the downstream stage parameters :
         - localizeWords Parameters
        """
        # Reset downstream functions
        self._resetLocalizeWordsParams()
        # > Parameters for localizing letters
        self.unpruned_num_cc: int = -1
        self.unpruned_image_cc_1C: np.ndarray = np.array([])
        self.unpruned_cc_stats: np.ndarray = np.array([])
        self.unpruned_cc_centroids: np.ndarray = np.array([])
        self.pruned_num_cc: int = -1
        self.pruned_image_cc_1C: np.ndarray = np.array([])
        self.pruned_cc_stats: np.ndarray = np.array([])
        self.pruned_cc_centroids: np.ndarray = np.array([])

        self.image_cc_3C_to_be_pruned: np.ndarray = np.array([])
        self.image_pruned_3C_letter_localized: np.ndarray = np.array([])
        self.image_original_letter_localized: np.ndarray = np.array([])
        self.image_original_masked_letter_localized: np.ndarray = np.array([])

        self.individual_letter_localized_edgeswt: np.ndarray = np.array([])
        self.individual_letter_localized_original: np.ndarray = np.array([])

        self.letter_stage_done: bool = False
        self.letter_min_done: bool = False
        self.letter_ext_done: bool = False
        self.letter_outline_done: bool = False

    # ######################################### #
    #             LOCALIZE WORDS                #
    # ######################################### #
    def _resetLocalizeWordsParams(self):
        """
        Resets the localizeWords stage parameters and the downstream stage parameters :
        """
        # > Parameters for localizing words
        self.image_pruned_3C_word_localized: np.ndarray = np.array([])
        self.image_original_word_localized: np.ndarray = np.array([])
        self.image_original_masked_word_localized: np.ndarray = np.array([])
        self.individual_word_localized_edgeswt: np.ndarray = np.array([])
        self.individual_word_localized_original: np.ndarray = np.array([])
        self.word_stage_done: bool = False
        self.word_bubble_done: bool = False
        self.word_polygon_done: bool = False
        self.word_bbox_done: bool = False
