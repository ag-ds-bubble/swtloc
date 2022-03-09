# Author : Achintya Gupta
# Purpose : Serves as an entry point

from .configs import config
from .abstractions import SWTImage
from .utils import SWTLocalizerValueError
from .configs import CONFIG__SWTLOCALIZER
from .utils import perform_type_sanity_checks
from .configs import TRANSFORM_INPUT__1C_IMAGE
from .configs import TRANSFORM_INPUT__3C_IMAGE
from .configs import CONFIG__SWTIMAGE__TRANSFORM
from .utils import generate_random_swtimage_names
from .configs import CONFIG__SWTLOCALIZER__MULTIPROCESSING
from .configs import CONFIG__SWTLOCALIZER__TRANSFORM_IMAGES
from .configs import CONFIG__SWTLOCALIZER__TRANSFORM_IMAGE_PATHS

import os
import numpy as np
from cv2 import cv2
from typing import List
from typing import Union
from copy import deepcopy
from typing import Optional


class SWTLocalizer:
    """
        ``SWTLocalizer`` acts as an entry point for performing Transformations and Localizations.
    It creates and houses a list of ``SWTImage`` objects in `swtimages` attribute, after
    sanity checks have been performed on the input given.

    The inputs can be a path (string) to an image file or an numpy array of the image.
    Inputs can also be just a single image filepath (string) or a single pre-loaded
    image (np.ndarray) or it could be a list of either image filepath or list of np.ndarray.

    But both the parameters i.e `image_paths` and `images` cannot be provided. Once the inputs
    provided to the ``SWTLocalizer`` class, sanity checks are performed on the input, and in case of
    `images` being provided, random numerical names are assigned to each image(s).

    Example:
    ::
        >>> # Import the SWTLocalizer class
        >>> from swtloc import SWTLocalizer
        >>> from cv2 import cv2
        >>>
        >>> root_path = 'examples/images/'
        >>>
        >>> # Single Image Path (NOTE : Use your own image paths)
        >>> single_image_path = root_path+'test_image_1/test_img1.jpg'
        >>> swtl = SWTLocalizer(image_paths=single_image_path)
        >>>
        >>> # Multiple Image Paths (NOTE : Use your own image paths)
        >>> multiple_image_paths = [root_path+'test_image_2/test_img2.jpg',
        >>>                         root_path+'test_image_3/test_img3.jpg',
        >>>                         root_path+'test_image_4/test_img4.jpeg' ]
        >>> swtl = SWTLocalizer(image_paths=multiple_image_paths)

        >>> # Single Pre-Loaded Image - Agnostic to image channels
        >>> single_image = cv2.imread(root_path+'test_image_1/test_img1.jpg')
        >>> swtl = SWTLocalizer(images=single_image)

        >>> # Multiple Pre-Loaded Image
        >>> multiple_images = [cv2.imread(each_path) for each_path in [root_path+'test_image_2/test_img2.jpg',
        >>>                                                            root_path+'test_image_3/test_img3.jpg',
        >>>                                                            root_path+'test_image_4/test_img4.jpeg' ]]
        >>> swtl = SWTLocalizer(images=multiple_images)

        >>> # Accessing `SWTImage` objects from the `SWTLocalizer`
        >>> multiple_images = [cv2.imread(each_path) for each_path in [root_path+'test_image_2/test_img2.jpg',
        >>>                                                            root_path+'test_image_3/test_img3.jpg',
        >>>                                                            root_path+'test_image_4/test_img4.jpeg' ]]
        >>> swtl = SWTLocalizer(images=multiple_images)
        >>> print(swtl.swtimages, type(swtl.swtimages[0]))
        [Image-SWTImage_982112, Image-SWTImage_571388, Image-SWTImage_866821] <class 'swtloc.abstractions.SWTImage'>

        >>> # Empty Initialisation -> Raises SWTLocalizerValueError (from v2.1.0)
        >>> swtl = SWTLocalizer()
        SWTLocalizerValueError: Either `images` or `image_paths` parameters should be provided.
        >>>
        >>>
        >>> # Mixed input given  -> Raises SWTLocalizerValueError
        >>> mixed_input = [root_path+'test_image_1/test_img1.jpg' , cv2.imread(root_path+'test_image_1/test_img1.jpg')]
        >>> swtl = SWTLocalizer(images=mixed_input)
        SWTLocalizerValueError: If a list is provided to `images`, each element should be an np.ndarray
        >>>
        >>> # Wrong input type given  -> Raises SWTLocalizerValueError
        >>> wrong_input = [True, 1, 'abc', root_path+'test_image_1/test_img1.jpg']
        >>> swtl = SWTLocalizer(image_paths=wrong_input)
        SWTLocalizerValueError: `image_paths` should be a `list` of `str`
        >>>
        >>>
        >>> # If the file is not present at the location (NOTE : Use your own image paths) -> Raises FileNotFoundError
        >>> multiple_image_paths = [root_path+'test_image_2/test_img2.jpg',
        >>>                         root_path+'test_image_/image_not_there.jpg',
        >>>                         root_path+'test_image_4/test_img4.jpeg' ]
        >>> swtl = SWTLocalizer(image_paths=multiple_image_paths)
        FileNotFoundError: No image present at ../swtloc/examples/test_images/image_not_there.jpg

        >>> # Random Names being assigned to each image when `images` parameter is provided
        >>> multiple_images = [cv2.imread(each_path) for each_path in [root_path+'test_image_2/test_img2.jpg',
        >>>                                                            root_path+'test_image_3/test_img3.jpg',
        >>>                                                            root_path+'test_image_4/test_img4.jpeg' ]]
        >>> swtl = SWTLocalizer(images=multiple_images)
        >>> print([each_image.image_name for each_image in swtl.swtimages])
        ['SWTImage_982112', 'SWTImage_571388', 'SWTImage_866821']
    """

    def __init__(self, multiprocessing: Optional[bool] = False,
                 images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                 image_paths: Optional[Union[str, List[str]]] = None):
        """
        Create a ``SWTLocalizer`` object which will house a list of ``SWTImage`` objects in
        `swtimage` attribute.

        Args:
            multiprocessing (Optional[bool]) : Whether to use multiprocessing or not. [default = False]
            images (Optional[Union[np.ndarray, List[np.ndarray]]]) : An individual image or a list of 3dimensional
             (RGB) or 1 dimensional (gray-scale)numpy array. [default = None]
            image_paths (Optional[Union[str, List[str]]]) : A single image path or a list of image paths.
             [default = None]
        Raises:
            - SWTLocalizerValueError
            - FileNotFoundError
        """
        # Variable Initialisation
        config[CONFIG__SWTLOCALIZER__MULTIPROCESSING] = multiprocessing

        config[CONFIG__SWTLOCALIZER__TRANSFORM_IMAGES] = images
        config[CONFIG__SWTLOCALIZER__TRANSFORM_IMAGE_PATHS] = image_paths
        self.swtimages: List[SWTImage] = []

        # Sanity Checks
        res_pack = self._sanityChecks(images=images, image_paths=image_paths)
        transform_inputs, transform_input_flags, transform_input_image_names = res_pack

        # Instantiate each transform_input as SWTImage
        for each_image, each_input_flag, each_image_name in zip(*[transform_inputs,
                                                                  transform_input_flags,
                                                                  transform_input_image_names]):
            swt_img_cfg = {k: v for k, v in config.items() if CONFIG__SWTIMAGE__TRANSFORM}
            swt_img_obj = SWTImage(image=each_image,
                                   image_name=each_image_name,
                                   input_flag=each_input_flag,
                                   cfg=deepcopy(swt_img_cfg))

            # Append the SWTImage object to swtimages list
            self.swtimages.append(swt_img_obj)

    @staticmethod
    def _sanityChecks(images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                      image_paths: Optional[Union[str, List[str]]] = None):
        """
        Perform sanity checks on `SWTLocalizer``'s input.

        Args:
            images (Optional[Union[np.ndarray, List[np.ndarray]]]) : An individual image or a list of 3
             dimensional (RGB) or 1 dimensional (gray-scale) numpy array.
            image_paths (Optional[Union[str, List[str]]]) : A single image path or a list of image paths.
        Returns:
            (List[np.ndarray]) : A list of images (numpy array). [default = None].
            (List[ByteString]) : A list of flags associated with each input image, one of
             [`TRANSFORM_INPUT__1C_IMAGE`, `TRANSFORM_INPUT__3C_IMAGE`], representing a single
             channel and a RGB channel image. [default = None].
            (List[str]) : A list of image names. In case of `image_paths` parameter being
             provided the names are taken from path of the image. In case of `images` parameter
             being provided, random image names are assigned to each image. [default = None].
        Raises:
            - SWTLocalizerValueError
            - FileNotFoundError
        """
        # Type Sanity checks
        perform_type_sanity_checks(cfg=config, cfg_of=CONFIG__SWTLOCALIZER)
        # Return Variables
        transform_inputs: List[np.ndarray] = []
        transform_input_flags: List[bytes] = []
        transform_input_image_names: List[str] = []

        # Either one of imgpaths & images should be given as input to the function
        if all([val is None for val in [images, image_paths]]) or all(
                [val is not None for val in [images, image_paths]]):
            raise SWTLocalizerValueError("Either `images` or `image_paths` parameters should be provided.")

        # If `image_paths` parameter is given
        if image_paths is not None:
            # If only a single file path is provided as a string.
            # Convert it into a list of images paths
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            elif isinstance(image_paths, list):
                if not all([isinstance(each_path, str) for each_path in image_paths]):
                    raise SWTLocalizerValueError("`image_paths` should be a `list` of `str`")

            for each_path in image_paths:
                if not os.path.isfile(each_path):
                    raise FileNotFoundError(f"No image present at {each_path}")
                else:
                    _img_name = " ".join(each_path.split('/')[-1].split('.')[:-1])
                    transform_input_image_names.append(_img_name)
                    transform_inputs.append(cv2.imread(each_path))

        # If transform_inputs have been populated from image_paths
        if transform_inputs:
            images = transform_inputs
            transform_inputs = []

        # If `images` parameter is given - np.ndarray, [np.ndarray]
        if images is not None:
            if isinstance(images, np.ndarray):
                images = [images]

            for each_image in images:
                # Check if all the elements in the list are images
                if not isinstance(each_image, np.ndarray):
                    raise SWTLocalizerValueError(
                        "If a list is provided to `images`, each element should be an np.ndarray")
                # Check if its whether 3d or 1d image
                if not (len(each_image.shape) in [3, 2]):
                    raise SWTLocalizerValueError(
                        "Every image in `images` parameter can only be 3D(RGB Image) or 2D(Grayscale Image)")

                if len(each_image.shape) == 3:
                    if each_image.shape[-1] != 3:
                        raise SWTLocalizerValueError(
                            "Every image in `images` parameter must be 3 Channels (RGB) image for a 3D image")
                    else:
                        transform_inputs.append(each_image)
                        transform_input_flags.append(TRANSFORM_INPUT__3C_IMAGE)
                elif len(each_image.shape) == 2:
                    transform_inputs.append(each_image)
                    transform_input_flags.append(TRANSFORM_INPUT__1C_IMAGE)

        # If the case was of `image_paths` then the image names have already been extracted
        # if not, then the case was that of the `images`
        if not transform_input_image_names:
            transform_input_image_names = generate_random_swtimage_names(n=len(transform_inputs))

        return transform_inputs, transform_input_flags, transform_input_image_names
