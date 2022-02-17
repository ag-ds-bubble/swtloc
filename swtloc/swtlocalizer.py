# Author : Achintya Gupta
# Purpose : Serves as an entry point

from ._swt import SWT
from ._utils import prepCC
from ._utils import imgshow
from ._utils import imgsave
from ._utils import prog_bar
from ._utils import auto_canny
from ._utils import print_valcnts
from ._bubble_bbox import BubbleBBOX

from .configs import config
from .abstractions import SWTImage
from .utils import deprecated_wrapper
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
import math
import time
import warnings
import numpy as np
from cv2 import cv2
from typing import List
from typing import Union
from copy import deepcopy
from typing import Optional

warnings.filterwarnings('ignore')

# TODO : To be removed in v2.1.0
COMPONENT_PROPS = {'bbm_h': None,
                   'bbm_w': None,
                   'bbm_cx': None,
                   'sw_var': None,
                   'bbm_cy': None,
                   'pixels': None,
                   'bbm_ar': None,
                   'bbm_ang': None,
                   'sw_mean': None,
                   'bbm_bbox': None,
                   'sw_median': None,
                   'bbm_outline': None,
                   'sw_countdict': None,
                   'img_color_mean': None,
                   'img_color_median': None}


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

        # TODO : To be removed in v2.1.0
        self.multiprocessing = multiprocessing
        self._obj_sanity_check_deprecated()
        # To facilitate (till v2.1.0) usage of empty object initialisation.
        # TODO : To be removed in v2.1.0
        if (images is None) and (image_paths is None):
            images = np.ones(shape=(10, 10))

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

    # TODO: To be removed in v2.1.0
    ##############################################
    #            DEPRECATED FUNCTIONS            #
    ##############################################

    def _obj_sanity_check_deprecated(self):
        if not isinstance(self.multiprocessing, bool):
            raise ValueError("Invalid 'multiprocessing' type, should be of type 'bool'")
        else:
            if self.multiprocessing:
                raise ValueError("Currently 'multiprocessing' mode is not available")

    @deprecated_wrapper(reason="Refactoring Changes", in_favour_of="'transform'", removed_in="v2.1.0")
    def swttransform(self, image=None, image_name='Trasnformed_Result', imgpaths=None,
                     save_results=False, save_rootpath='../SWTLocResults/', *args, **kwargs):

        """
        Function to apply SWT transform on the images present in the imgpaths

        parameters
        --------------------------------------
        image : numpy.ndarray,
            Either a 3 channel(RGB) or 1 Channel(GrayScale) image as an input.

        imgpaths : str or list, required
            Path of all the images to be transformed.
        
        save_results : bool, optional, default : False
            Whether to save the results.

        save_rootpath : str, optional, default : '../SWTLocResults/'
            Base path to the save path

        text_mode : str, optional, default : 'lb_df'
            Contrast of the text present in the image, which needs to be
            transformed. Two possible values :

                a.) db_lf :> Dark Background Light Foreground i.e Light color text on Dark color background
                2.) lb_df :> Light Background Dark Foreground i.e Dark color text on Light color background

            This parameters affect how the gradient vectors (the direction) 
            are calulated, since gradient vectors of db_lf are in  −𝑣𝑒  direction 
            to that of lb_df gradient vectors.

        gs_blurr : bool, optional, default : True
            Right after an image is read from the path provided, the image 
            will be converted into grayscale (default). To contol this 
            operation following parameter can be used :-

                gs_blurr = True :> Wether to blurr an image or not.
                blurr_kernel = (5,5) [If gs_blurr = True] :> The kernel size for the operation

        edge_func : str or callable, optional, default : 'ac'
            Finding the Edge of the image is a tricky part, this is pertaining
            to the fact that in most of the cases the images we deal with are not
            of standard that applying just a opencv Canny operator would
            result in the desired Edge Image.

            Sometimes (In most cases) there is some custom processing required
            before edging, for that reason alone this function accepts one of
            the following two values :-

                1.) 'ac' :> Auto-Canny function, an in-built function which will
                            generate the Canny Image from the original image, internally
                            calculating the threshold parameters, although, to tune it
                            even further 'ac_sigma : float, default(0.33)' parameter is provided which
                            can take any value between 0.0 <--> 1.0.

                2.) A custom function : This function should have its signature as mentioned below :

                        def custom_edge_func(gray_image):
                            Your Function Logic...
                            edge_image = ...
                            return edge_image

        minrsw : float, optional, default : 3
            Minimum Stroke Width

        maxrsw : float, optional, default : 200
            Maximum Stroke Width

        check_anglediff : bool, optional, default : True
            Wether to check the Wether to check the angle deviation of originating
            edge pixel to the final resting pixel

        max_angledev : float, optional, default : π/6
             Permissible value of the angle deviation

        minCC_comppx : int, optional, default : 50
            Pruning Paramter : Minimum number of pixels to reside within each CC.

        maxCC_comppx : int, optional, default : 10000
            Pruning Paramter : Maximum number of pixels to reside within each CC.

        acceptCC_aspectratio : float, default : 5
            Pruning Paramter : Acceptable Inverse of the Aspect Ratio of each CC.


        Attributes
        ---------------------------------------
            - orig_img : Original image, np.ndarray
            - grayedge_img : Gray Scale Edged Image, np.ndarray
            - img_gradient : Gradient Theta of the Edged Image, np.ndarray
            - swt_mat : Result of Stroke Width Transform on the image
            - swt_labelled : Connected Components labelled mask for the swt_mat, Monochromatic.
            - swt_labelled3C : Connected Components labelled mask for the swt_mat, RGB channel
            - swtlabelled_pruned1 : Pruned Connected Components labelled mask for the swt_mat, Monochromatic.
            - swtlabelled_pruned13C : Pruned Connected Components labelled mask for the swt_mat, RGB channel
            - transform_time : Time taken during this transformation.

        """
        # Initialisation
        self.image = image
        self.image_name = image_name
        self.imgpaths = imgpaths
        # Image Type Flags
        self._individual_image_input = False
        self._individual_image_3c = False
        # Prepare the save root path
        self.save_rootpath = os.path.abspath(save_rootpath)
        self._sanity_check_transform_deprecated(kwargs)

        if self.imgpaths is not None:
            # Progress bar to report the total done
            self._probar = prog_bar(self.imgpaths, len(self.imgpaths))
            for each_imgpath in self._probar:
                self.components_props = {}
                self._transform_deprecated(imgpath=each_imgpath, **kwargs)
                if save_results:
                    imgname = each_imgpath.split('/')[-1].split('.')[0]
                    savepath = self.save_rootpath + f'/{imgname}'
                    self._save_deprecated(savepath=savepath)
        else:
            print('Transforming...')
            self.components_props = {}
            self._transform_deprecated(image=self.image, **kwargs)
            if save_results:
                savepath = self.save_rootpath + f'/{self.image_name}'
                self._save_deprecated(savepath=savepath)
            print('Operation Completed!')

    def _save_deprecated(self, savepath):
        """
        Function to save swtransform results to the savepath. Folder
        path will be determined by the name of the image, so make sure
        the image names are unique.

        parameters
        --------------------------------------
        savepath : str, required
            Save location path

        Saves:
            Original @ savepath+'/orig_img.png' : Original Image
            EdgeImage @ savepath+'/edge_img.png' : Edged Image
            Gradient @ savepath+'/grad_img.png' : Gradient Theta
                                                  Image
            SWT Transform @ savepath+'/swt3C_img.png' : Connected
                                                        Components after
                                                        SWT Transform
            SWT Pruned @ savepath+'/swtpruned3C_img.png' : Connected
                                                           Components after
                                                           SWT Transform and Pruning.
        """
        os.makedirs(savepath, exist_ok=True)

        # Save Original Image
        imgsave(self.orig_img, title='Original', savepath=savepath + '/orig_img.png')

        # Save Gray Edge Image
        imgsave(self.grayedge_img, title='EdgeImage', savepath=savepath + '/edge_img.png')

        # Save Gradient Image
        imgsave(self.img_gradient, title='Gradient', savepath=savepath + '/grad_img.png')

        # Save Labelled 3 Channel Image
        imgsave(self.swt_labelled3C, title='SWT Transform', savepath=savepath + '/swt3C_img.png')

        # Save Pruned Labelled 3 Channel Image
        imgsave(self.swtlabelled_pruned13C, title='SWT Pruned', savepath=savepath + '/swtpruned3C_img.png')

    def _sanity_check_transform_deprecated(self, kwargs):
        """
        Sanity Check for swtransform parameters
        """
        # Either one of imgpaths & image should be given as input to the function
        if self.imgpaths is not None:
            # Check for imgpaths
            if isinstance(self.imgpaths, str) or isinstance(self.imgpaths, list):
                if isinstance(self.imgpaths, list):
                    self.progress_ind = False
                    for eachPath in self.imgpaths:
                        if not os.path.isfile(eachPath):
                            raise FileNotFoundError(f"No image present at {eachPath}")

                if isinstance(self.imgpaths, str):
                    if not os.path.isfile(self.imgpaths):
                        raise FileNotFoundError(f"No image present at {self.imgpaths}")
                    self.imgpaths = [self.imgpaths]  # Convert a single image path to a list
            else:
                raise ValueError("'imgpaths' argument needs to be of type 'str' or 'list'")
        elif self.image is not None:
            # Check if its an ndarray
            if not isinstance(self.image, np.ndarray):
                raise ValueError(
                    "When provinding a singular image through 'image' argument, the value should be np.ndarray with 3 Channels(RGB) or 1 Channel(GrayScale), i.e the output of cv2.imread()")
            # Check if its wether 3d or 1d image
            if not (len(self.image.shape) in [3, 2]):
                raise ValueError(
                    "When provinding a singular image through 'image' argument, the value should be np.ndarray with 3 Channels(RGB) or 1 Channel(GrayScale), i.e the output of cv2.imread()")
            # Check for the number of channels for a 3 channel image
            if len(self.image.shape) == 3:
                if self.image.shape[-1] != 3:
                    raise ValueError(
                        "When provinding a singular image through 'image' argument, the value should be np.ndarray with 3 Channels(RGB) or 1 Channel(GrayScale), i.e the output of cv2.imread()")
                elif self.image.shape[-1] == 3:
                    self._individual_image_3c = True  # Update Falgs
            self._individual_image_input = True  # Update Falgs
        else:
            raise ValueError("Either one of 'imgpaths' or 'image' argument needs to be provided")

        if not isinstance(self.image_name, str):
            raise ValueError("'image_name' should be of type `str`")

        # Save the savepath directory, if not there then make one
        if not os.path.isdir(self.save_rootpath):
            os.makedirs(self.save_rootpath, exist_ok=True)

        # Check for the kwargs for the transform function
        if 'text_mode' in kwargs:
            if not (kwargs['text_mode'] in ['db_lf', 'lb_df']):
                raise ValueError("'text_mode' should be one of ['db_lf', 'lb_df']")
        if 'gs_blurr' in kwargs:
            if not isinstance(kwargs['gs_blurr'], bool):
                raise ValueError("'gs_blurr' should be of type bool")
        if 'blurr_kernel' in kwargs:
            if not (isinstance(kwargs['blurr_kernel'], tuple) and all(
                    isinstance(k, int) and (k % 2 != 0) and (k >= 3) for k in kwargs['blurr_kernel']) and (
                            kwargs['blurr_kernel'][0] == kwargs['blurr_kernel'][1])):
                raise ValueError("'blurr_kernel' should be of type tuple, and must contain integer odd values")
        if 'edge_func' in kwargs:
            if isinstance(kwargs['edge_func'], str) or callable(kwargs['edge_func']):
                if isinstance(kwargs['edge_func'], str):
                    if not (kwargs['edge_func'] in ['ac']):
                        raise ValueError("'edge_func' should be one of ['ac']")
                elif not callable(kwargs['edge_func']):
                    raise ValueError("'edge_func' custom function which returns an edged image.")
            else:
                raise ValueError("'edge_func' should be either 'ac' or callable")
        if 'ac_sigma' in kwargs:
            if not (isinstance(kwargs['ac_sigma'], float) and (0.0 <= kwargs['ac_sigma'] <= 1.0)):
                raise ValueError("'ac_sigma' should be of type float and value between 0 and 1")
        if 'minrsw' in kwargs:
            if not (isinstance(kwargs['minrsw'], int) and kwargs['minrsw'] >= 3):
                raise ValueError("'minrsw' should be of type int and be more than 3")
        if 'maxrsw' in kwargs:
            if not (isinstance(kwargs['maxrsw'], int) and kwargs['maxrsw'] >= 3):
                raise ValueError("'maxrsw' should be of type int and be more than 3")
        if ('minrsw' in kwargs) and ('maxrsw' in kwargs):
            if kwargs['maxrsw'] <= kwargs['minrsw']:
                raise ValueError("'minrsw' should be smaller than 'maxrsw'")
        if 'max_angledev' in kwargs:
            if not (isinstance(kwargs['max_angledev'], float) and (-np.pi / 2 <= kwargs['max_angledev'] <= np.pi / 2)):
                raise ValueError("'max_angledev' should be a float and inbetween -90° <-> 90° (in radians)")
        if 'check_anglediff' in kwargs:
            if not (isinstance(kwargs['check_anglediff'], bool)):
                raise ValueError("'isinstance' should be type bool")
        if 'minCC_comppx' in kwargs:
            if not (isinstance(kwargs['minCC_comppx'], int) and kwargs['minCC_comppx'] > 0):
                raise ValueError("'minCC_comppx' should be type int")
        if 'maxCC_comppx' in kwargs:
            if not (isinstance(kwargs['maxCC_comppx'], int) and kwargs['maxCC_comppx'] > 0):
                raise ValueError("'maxCC_comppx' should be of type int")
        if 'acceptCC_aspectratio' in kwargs:
            if not (isinstance(kwargs['acceptCC_aspectratio'], float) and kwargs['acceptCC_aspectratio'] > 0.0):
                raise ValueError("'acceptCC_aspectratio' should be of type float and positive")

    def _transform_deprecated(self, image=None, imgpath=None, text_mode='lb_df',
                              gs_blurr=True, blurr_kernel=(5, 5),
                              edge_func='ac', ac_sigma=0.33,
                              minrsw=3, maxrsw=200, max_angledev=np.pi / 6, check_anglediff=True,
                              minCC_comppx=50, maxCC_comppx=10000, acceptCC_aspectratio=5):
        """
        Entry Point for the Stroke Width Transform - Single Image
        """

        ts = time.perf_counter()

        # Read the image..
        if image is None:
            self.orig_img, origgray_img = self._image_read_deprecated(imgpath=imgpath, gs_blurr=gs_blurr,
                                                                      blurr_kernel=blurr_kernel)
        elif imgpath is None:
            self.orig_img, origgray_img = self._image_read_deprecated(image=image, gs_blurr=gs_blurr,
                                                                      blurr_kernel=blurr_kernel)

        # Find the image edge
        origgray_img, self.grayedge_img = self._image_edge_deprecated(gray_image=origgray_img, edge_func=edge_func,
                                                                      ac_sigma=ac_sigma)

        # Find the image gradient
        self.img_gradient = self._image_gradient_deprecated(original_img=origgray_img, edged_img=self.grayedge_img)
        hstep_mat = np.round(np.cos(self.img_gradient), 5)
        vstep_mat = np.round(np.sin(self.img_gradient), 5)
        if text_mode == 'db_lf':
            hstep_mat *= -1
            vstep_mat *= -1

        # Find the Stroke Widths in the Image
        self._swtObj = SWT(edgegray_img=self.grayedge_img, hstepmat=hstep_mat, vstepmat=vstep_mat,
                           imggradient=self.img_gradient,
                           minrsw=minrsw, maxrsw=maxrsw, max_angledev=max_angledev, check_anglediff=check_anglediff)
        self.swt_mat = self._swtObj.find_strokes()

        # Find the connected Components in the image
        numlabels, self.swt_labelled = self._image_swtcc_deprecated(swt_mat=self.swt_mat)
        self.swt_labelled3C = prepCC(self.swt_labelled)

        # Prune and Extract LabelComponent Properties
        self.swtlabelled_pruned1 = self._image_prune_getprops_deprecated(orig_img=self.orig_img,
                                                                         swtlabelled=self.swt_labelled,
                                                                         minCC_comppx=minCC_comppx,
                                                                         maxCC_comppx=maxCC_comppx,
                                                                         acceptCC_aspectratio=acceptCC_aspectratio)
        self.swtlabelled_pruned13C = prepCC(self.swtlabelled_pruned1)

        self.transform_time = str(np.round(time.perf_counter() - ts, 3)) + ' sec'

    def _image_read_deprecated(self, image=None, imgpath=None, gs_blurr=True, blurr_kernel=(5, 5)):
        """
        Function to read image from imgpath, covert to grayscale
        apply Gaussian Blurr based on the arguments .

        parameters
        --------------------------------------
        imgpath : str, required
            Path to the image, single Image.

        gs_blurr : bool, optional, default : True
            Wether to apply Gaussain Blurr after the image has been read

        blurr_kernel : tuple, default : (5,5)
            If gs_blurr = True, then size of the kernel with which it needs
            convolve with.

        returns
        --------------------------------------
        tuple - (orig_img, origgray_img)

        Returns Original Image and Grayscale Image
        """
        if image is None:
            orig_img = cv2.imread(imgpath)  # Read Image
            origgray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
        else:
            orig_img = image
            if self._individual_image_3c == True:
                origgray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
            else:
                origgray_img = orig_img
        if gs_blurr:
            origgray_img = cv2.GaussianBlur(origgray_img, blurr_kernel, 0)

        return orig_img, origgray_img

    @staticmethod
    def _image_edge_deprecated(gray_image, edge_func, ac_sigma=0.33):
        """
        Function to find edge of the image.

        parameters
        --------------------------------------
        gray_image : np.ndarray, required
            Gray Scale Image

        edge_func : str or callable, optional, default : 'ac'
            Edge function, function to be used to find the
            edge of an image. It can take one of these two as
            inputs :-
                1.) 'ac' - str, Auto Canny function
                2.) custom_func - Callable, A Custom Function
                    which has a following signature

                    def custom_edge_func(gray_image):

                        # Your Function Logic...
                        edge_image = ...

                        return edge_image

        ac_sigma : float, default : 0.33
            Thresholding parameter when edge_func = 'ac', .

        returns
        --------------------------------------
        tuple - (gray_image, image_edge)

        Returns Grayscale Image and Edged Image
        """
        image_edge = None
        if edge_func == 'ac':
            image_edge = auto_canny(image=gray_image, sigma=ac_sigma)
        elif callable(edge_func):
            try:
                image_edge = edge_func(gray_image)
            except:
                pass

        return gray_image, image_edge

    @staticmethod
    def _image_gradient_deprecated(original_img, edged_img):
        """
        Function to find gradient vectors (θ) of the image.
        Using sobel operators to find the  gradient in dx
        and dy direction using 5x5 kernel θ matrix is formed
        and only those indexex are retained where the edged_img!=0.

        parameters
        --------------------------------------
        original_img : np.ndarray, required
            Orignal Grayscale Image

        edged_img : np.ndarray, required
            Edge image of  Grayscale Image

        returns
        --------------------------------------
        tuple - (gray_image, image_edge)

        Returns Grayscale Image and Edged Image
        """
        rows, columns = original_img.shape[:2]
        dx = cv2.Sobel(original_img, cv2.CV_32F, 1, 0, ksize=5, scale=-1, delta=1, borderType=cv2.BORDER_DEFAULT)
        dy = cv2.Sobel(original_img, cv2.CV_32F, 0, 1, ksize=5, scale=-1, delta=1, borderType=cv2.BORDER_DEFAULT)

        theta_mat = np.arctan2(dy, dx)
        edgesbool = (edged_img != 0).astype(int)
        theta_mat = theta_mat * edgesbool

        return theta_mat

    @staticmethod
    def _image_swtcc_deprecated(swt_mat):
        """
        Function to find connected components of the SWT image,
        each connected component region is filled with a unique label
        value.

        parameters
        --------------------------------------
        swt_mat : np.ndarray, required
            SWT Transform of the image

        returns
        --------------------------------------
        tuple - (num_labels, labelmask)

        Returns Number of lables found and nd.array for the
        of the connected components.
        """
        threshmask = swt_mat.copy().astype(np.int16)
        threshmask[threshmask == np.max(
            threshmask)] = 0  # Set the maximum value(Diagonal of the Image :: Maximum Stroke Width) to 0
        threshmask[threshmask > 0] = 1
        threshmask = threshmask.astype(np.int8)

        num_labels, labelmask = cv2.connectedComponents(threshmask, connectivity=8)

        return num_labels, labelmask

    def _image_prune_getprops_deprecated(self, orig_img, swtlabelled, minCC_comppx, maxCC_comppx, acceptCC_aspectratio):
        """
        Function to find Prune the Connected Component Labelled image. Based on
        the parameters values the Connected Component mask will be pruned and
        and certain properties will be calculated against each Component,
        as mentioned below :

        # Note
        *CC :Connected Component
        *bbm : Bounding Box Mininum. This is a result to cv2.minAreaRect function,
               which would return the minimum area rectangle bounding box for a
               CC
        *sw : Stroke Width within that component

            pixels : Number of pixels whithin a particular CC
            bbm_h : Minimum Bounding Box Height
            bbm_w : Minimum Bounding Box Width
            bbm_cx : Minimum Bounding Box Centre x
            bbm_cy : Minimum Bounding Box Cntre y
            bbm_ar : Minimum Bounding Box Aspect Ratio (bbm_w/bbm_h)
            bbm_bbox : Cooridinates of the bbm vertices
            bbm_anchor : Vertice corresponding to least x coord
            bbm_outline : BBM outline, i.e outer contour
            bbm_ang : Angle of orientation for that BBM. Both bbm_ang and 180-bbm_ang
                      can be valid
            img_color_mean : Mean (R,G,B) tuple values of the original image
                             masked by that component
            img_color_median : Median (R,G,B) tuple values of the original image
                               masked by that component
            sw_countdict : Value Counts for the different stroke widths within that
                           CC
            sw_var : Stroke Width variance within each CC
            sw_median : Median stroke Width within each CC
            sw_mean : Mean stroke Width within each CC

        PRUNE PARAMETERS : minCC_comppx, maxCC_comppx, acceptCC_aspectratio
        OTHER PARAMETERS : orig_img, swtlabelled


        parameters
        --------------------------------------
        orig_img : nd.ndarray, required
            Original Image ndarray.

        swtlabelled : nd.ndarray, required
            Connected Components labelled mask after SWT.

        minCC_comppx : int, optional, default : 50
            Pruning Paramter : Minimum number of pixels to reside within each CC.

        minCC_comppx : int, optional, default : 10000
            Pruning Paramter : Maximum number of pixels to reside within each CC.

        acceptCC_aspectratio : float, default : 5
            Pruning Paramter : Acceptable Inverse of the Aspect Ratio of each CC.

        returns
        --------------------------------------
        nd.ndarray - swtlabelled_pruned

        Returns Pruned SWT Labelled Image
        """
        swtlabelled_pruned = swtlabelled.copy()
        lc_count = print_valcnts(swtlabelled_pruned, _print=False)
        # Pruning based on min and max number of pixels in a connected component
        for label, count in lc_count.items():
            if count < minCC_comppx or count > maxCC_comppx:
                swtlabelled_pruned[swtlabelled_pruned == label] = 0
        lc_count = print_valcnts(swtlabelled_pruned, _print=False)

        # Pruning based on a Aspect Ratio
        for label, pixel_count in lc_count.items():

            lmask = (swtlabelled_pruned == label).astype(np.uint8).copy()

            cntrs = cv2.findContours(lmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            rotrect = cv2.minAreaRect(cntrs[0])
            label_height = np.round(max(rotrect[1]), 2)
            label_width = np.round(min(rotrect[1]), 2)
            label_aspectratio = label_width / label_height

            if not ((1 / acceptCC_aspectratio) < label_aspectratio < acceptCC_aspectratio):
                swtlabelled_pruned[swtlabelled_pruned == label] = 0
            else:
                bbm_cx, bbm_cy = np.round(rotrect[0], 2)
                bbm_bbox = cv2.boxPoints(rotrect)

                anchor_point = bbm_bbox[np.argmax((bbm_bbox == np.min(bbm_bbox[:, 0])).sum(axis=1))]
                remain_point = np.array([k for k in bbm_bbox if (k != anchor_point).any()])
                all_lengths = [np.linalg.norm(k - anchor_point) for k in remain_point]
                anchor_armlength_point = remain_point[all_lengths == np.sort(all_lengths)[1]][0]

                bbox_ang = np.arctan(
                    -(anchor_armlength_point[1] - anchor_point[1]) / (anchor_armlength_point[0] - anchor_point[0]))
                bbox_ang = np.rad2deg(bbox_ang)
                if bbox_ang < 0:
                    bbox_ang = 180 + bbox_ang

                self.components_props[label] = COMPONENT_PROPS.copy()
                self.components_props[label]['pixels'] = pixel_count
                self.components_props[label]['bbm_h'] = label_height
                self.components_props[label]['bbm_w'] = label_width
                self.components_props[label]['bbm_cx'] = bbm_cx
                self.components_props[label]['bbm_cy'] = bbm_cy
                self.components_props[label]['bbm_ar'] = label_aspectratio
                self.components_props[label]['bbm_bbox'] = bbm_bbox
                self.components_props[label]['bbm_anchor'] = anchor_point
                self.components_props[label]['bbm_outline'] = cntrs
                self.components_props[label]['bbm_ang'] = bbox_ang

                _iy, _ix = lmask.nonzero()
                mean_rgbcolor = self.orig_img[_iy, _ix].mean(axis=0)
                median_rgbcolor = np.median(self.orig_img[_iy, _ix], axis=0)

                if median_rgbcolor.shape == () and mean_rgbcolor.shape == ():
                    self.components_props[label]['img_color_mean'] = str([np.floor(mean_rgbcolor)])
                    self.components_props[label]['img_color_median'] = str([np.floor(median_rgbcolor)])
                else:
                    self.components_props[label]['img_color_mean'] = str(np.floor(mean_rgbcolor).tolist())
                    self.components_props[label]['img_color_median'] = str(np.floor(median_rgbcolor).tolist())

                sw_xyvals = self.swt_mat[_iy, _ix].copy()
                sw_countdict = print_valcnts(sw_xyvals, _print=False, remove_0=False)

                self.components_props[label]['sw_countdict'] = str(sw_countdict)
                self.components_props[label]['sw_var'] = np.var(sw_xyvals)
                self.components_props[label]['sw_median'] = np.median(sw_xyvals)
                self.components_props[label]['sw_mean'] = np.mean(sw_xyvals)

        return swtlabelled_pruned

    # Get the Grouping and Bounding BBoxes
    @deprecated_wrapper(reason="Refactoring Changes", in_favour_of="'SWTImage.localizeLetters'",
                        removed_in="v2.1.0", relocated=True)
    def get_min_bbox(self, show=False, padding=5):
        """
        Function to retrieve the vetrices of BBox which occupy minimum
        area rectangle for a Connected Component.

        parameters
        --------------------------------------
        show : bool, optional, default : True
            Wether to show the annotated image with min_bboxes

        padding : int, optional, default : 5
            Expansion coefficient (in the diagonal direction) for each
            bbox

        returns
        --------------------------------------
        tuple - (min_bboxes, annotated_img)

        Returns Minimum Area BBoxes vertices and Annotated Image
        """
        if not hasattr(self, 'swt_mat'):
            raise Exception("Call 'swttransform' on the image before calling this function")

        min_bboxes = []
        annotated_img = self.swtlabelled_pruned13C.copy()

        for label, labelprops in self.components_props.items():
            bbm_bbox = np.int32(labelprops['bbm_bbox'])

            # Calculate centre coordinates
            _tr, _br, _bl, _tl = bbm_bbox.copy()
            _d1_vec = _tr - _bl
            _d2_vec = _tl - _br
            _d1_ang = -math.atan2(_d1_vec[1], _d1_vec[0])
            _d2_ang = -math.atan2(_d2_vec[1], _d2_vec[0])

            _tr = _tr + padding * np.array([np.cos(_d1_ang), -np.sin(_d1_ang)])
            _br = _br - padding * np.array([-np.cos(np.pi - _d2_ang), -np.sin(np.pi - _d2_ang)])
            _bl = _bl - padding * np.array([-np.cos(np.pi - _d1_ang), -np.sin(np.pi - _d1_ang)])
            _tl = _tl + padding * np.array([np.cos(_d2_ang), -np.sin(_d2_ang)])
            bbm_bbox = np.c_[_tr, _br, _bl, _tl].T.astype(int)

            min_bboxes.append(bbm_bbox)
            annotated_img = cv2.polylines(annotated_img, [bbm_bbox], True, (0, 0, 255), 1)

        if show:
            imgshow(annotated_img, 'Minimum Bounding Box')

        return min_bboxes, annotated_img

    @deprecated_wrapper(reason="Refactoring Changes", in_favour_of="'SWTImage.localizeLetters'",
                        removed_in="v2.1.0", relocated=True)
    def get_extreme_bbox(self, show=False, padding=5):
        """
        Function to retrieve the vetrices of BBox Connected Component.

        parameters
        --------------------------------------
        show : bool, optional, default : True
            Wether to show the annotated image with extreme_bboxes

        padding : int, optional, default : 5
            Expansion coefficient for each bbox

        returns
        --------------------------------------
        tuple - (ext_bboxes, annotated_img)

        Returns BBoxes vertices and Annotated Image
        """

        if not hasattr(self, 'swt_mat'):
            raise Exception("Call 'swttransform' on the image before calling this function")

        ext_bboxes = []
        temp1 = self.swtlabelled_pruned1.copy()
        annotated_img = self.swtlabelled_pruned13C.copy()

        for label, labelprops in self.components_props.items():
            lmask = (temp1 == label).astype(np.uint16)
            if np.sum(lmask) > 0:
                _iy, _ix = lmask.nonzero()
                _tr = [max(_ix) + padding, min(_iy) - padding]
                _br = [max(_ix) + padding, max(_iy) + padding]
                _bl = [min(_ix) - padding, max(_iy) + padding]
                _tl = [min(_ix) - padding, min(_iy) - padding]
                bbe_bbox = np.c_[_tr, _br, _bl, _tl].T.astype(int)
                ext_bboxes.append(bbe_bbox)

                annotated_img = cv2.polylines(annotated_img, [bbe_bbox], True, (0, 0, 255), 1)

        if show:
            imgshow(annotated_img, 'Extreme Bounding Box')

        return ext_bboxes, annotated_img

    @deprecated_wrapper(reason="Refactoring Changes", in_favour_of="'SWTImage.localizeLetters'",
                        removed_in="v2.1.0", relocated=True)
    def get_comp_outline(self, show=False):
        """
        Function to retrieve the outer contour of Connected Component.

        parameters
        --------------------------------------
        show : bool, optional, default : True
            Wether to show the annotated image with outline
        
        returns
        --------------------------------------
        tuple - (outlines, annotated_img)

        Returns outline and Annotated Image
        """

        if not hasattr(self, 'swt_mat'):
            raise Exception("Call 'swttransform' on the image before calling this function")

        outlines = []
        temp = self.swtlabelled_pruned13C.copy()
        for label, labelprops in self.components_props.items():
            loutline = labelprops['bbm_outline']
            outlines.append(loutline)

            temp = cv2.polylines(temp, loutline, True, (0, 0, 255), 1, 4)

        if show:
            imgshow(temp, 'Component Outlines')

        return outlines, temp

    @deprecated_wrapper(reason="Refactoring Changes", in_favour_of="'SWTImage.localizeWords'",
                        removed_in="v2.1.0", relocated=True)
    def get_grouped(self, lookup_radii_multiplier=0.8, sw_ratio=2.0,
                    cl_deviat=[13, 13, 13], ht_ratio=2.0, ar_ratio=3.0, ang_deviat=30.0,
                    bubble_width=1):
        """
        Function to Group Connected Component into possible 'words' based 
        on argument value.

        parameters
        --------------------------------------
        lookup_radii_multiplier : float, optional, default : 0.8
            lookup_radius = Connected Component Length * lookup_radii_multiplier

            Radius 'radius' to be looked up for grouping, for each Connected Component

        sw_ratio : float, optional, default : 2.0
            Acceptable ratio of the Stroke Width between components, for
            Grouping.

        cl_deviat : list, optional, default : [13,13,13]
            Acceptable difference [R,G,B] between median connected component
            colour, for Grouping.

        ht_ratio : float, optional, default : 2.0
            Acceptable ratio of the Height between components, for Grouping.

        ar_ratio : float, optional, default : 0.8
            Acceptable inverse of ascpect ratio between components, for Grouping.
        
        ang_deviat : float, optional, default : 0.8
            Acceptable deviation in orientation angle between components, for Grouping.
        
        returns
        --------------------------------------
        tuple - (grouped_labels, grouped_bubblebbox, grouped_annot_bubble, grouped_annot, maskviz, maskcomb)

            grouped_labels : List of grouped labels
            grouped_bubblebbox : List of Bubble BBox Contours
            grouped_annot_bubble : Annotated Bubble BBox
            grouped_annot : Annotated BBox
            maskviz : BBox and Label Visualisation
            maskcomb : Circular mask to each BBox with lookup_radii
        """

        if not hasattr(self, 'swt_mat'):
            raise Exception("Call 'swttransform' on the image before calling this function")

        bubbleBbox = BubbleBBOX(labelmask=self.swtlabelled_pruned1, comp_props=self.components_props,
                                lookup_radii_multiplier=lookup_radii_multiplier,
                                sw_ratio=sw_ratio, cl_deviat=cl_deviat, ht_ratio=ht_ratio, ar_ratio=ar_ratio,
                                ang_deviat=ang_deviat, bubble_width=bubble_width)
        grouped_labels, grouped_bubblebbox, grouped_annot_bubble, grouped_annot, maskviz, maskcomb = bubbleBbox.run_grouping()

        return grouped_labels, grouped_bubblebbox, grouped_annot_bubble, grouped_annot, maskviz, maskcomb
