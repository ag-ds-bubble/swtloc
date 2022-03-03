# Author : Achintya Gupta
# Purpose : Houses Core Algorithms for Stroke Width Transforms

from typing import List
import numpy as np
import numba as nb


def swt_strokes(edged_image,
                hstep_mat,
                vstep_mat,
                dstep_mat,
                max_stroke_width,
                min_stroke_width,
                image_height,
                image_width,
                check_angle_deviation,
                image_gradient_theta,
                max_angle_deviation,
                include_edges_in_swt):
    """
    Core Logic for Stroke Width Transform.
    Implementing the work of [Boris Epshtein, Eyal Ofek & Yonatan Wexler](https://www.microsoft.com/en-us/research/publication/detecting-text-in-natural-scenes-with-stroke-width-transform/)

    Objective of this function is to, given an edged input image, find the stroke widths conforming
    to the following rules :
        - Each Stroke Width has be in the range of : min_stroke_width<= stroke_widths<=max_stroke_width
        - A ray emanating from each edge point, traveling in its gradients direction, when met with another
        edge point will terminate its journey only when the difference between their gradient directional angles
        is np.pi - max_angle_deviation <= theta_diff <= np.pi + max_angle_deviation

    Args:
        edged_image (np.ndarray) : Edges of the Original Input Image. Same size as the original image

        hstep_mat (np.ndarray) : For each pixel, cos(gradient_theta), where gradient_theta is the gradient
         angle for that pixel, representing length of horizontal movement for every unit movement in gradients direction.
         Same size as the original image

        vstep_mat (np.ndarray) : For each pixel, sin(gradient_theta), where gradient_theta is the gradient
         angle for that pixel, representing length of vertical movement for every unit movement in gradients direction.
         Same size as the original image

        dstep_mat (np.ndarray) : np.sqrt(hstep_mat**2+vstep_mat**2)

        max_stroke_width (int) : Maximum Stroke Width which would be permissible

        min_stroke_width (int) : Minimum Stroke Width which would be required

        image_height (int) : Height of the image

        image_width (int) : Width of the image

        check_angle_deviation (bool) : Whether to check the angle deviation to terminate the ray

        image_gradient_theta (np.ndarray) : Gradient array of the input image

        max_angle_deviation (float) : Maximum Angle Deviation which would be permissible

        include_edges_in_swt (bool) : Whether to include edges in the final SWT result

    Returns:
        (np.ndarray) : Stroke Width Transformed Image, each stroke filled with stroke length.
    """
    # Initialisations
    edge_y, edge_x = edged_image.nonzero()
    edge_indices_set = set(zip(edge_y, edge_x))
    angle_dev_ll = np.pi - max_angle_deviation
    angle_dev_ul = np.pi + max_angle_deviation
    if include_edges_in_swt:
        swt_matrix = edged_image.copy()
    else:
        swt_matrix = np.zeros(shape=edged_image.shape, dtype=np.int32)
    ray_pointer = 0
    ray_length = -1
    ray_indices = np.full(shape=(max_stroke_width, 2), fill_value=np.nan, dtype=np.int32)

    for iy, ix in edge_indices_set:
        ray_pointer = 0

        # Get the starting indices and the step values
        delx = hstep_mat[iy, ix]
        dely = vstep_mat[iy, ix]
        deld = dstep_mat[iy, ix]
        itheta = image_gradient_theta[iy, ix]

        # Add the first point and increment the ray pointer
        ray_indices[ray_pointer] = [iy, ix]
        ray_pointer += 1
        breach = False

        while not breach:
            # Get the next point in the gradient direction
            niy = np.int32(np.floor(iy + ray_pointer * dely))
            nix = np.int32(np.floor(ix + ray_pointer * delx))

            ray_length = ray_pointer * deld

            max_sw_check = ray_length <= max_stroke_width - 2
            boundary_check = (0 <= niy < image_height) and (0 <= nix < image_width)
            edge_indices_check = (niy, nix) not in edge_indices_set

            if not (max_sw_check and boundary_check and edge_indices_check):
                if not edge_indices_check:
                    if check_angle_deviation:
                        theta_diff = np.abs(itheta - image_gradient_theta[niy, nix])
                        angle_check = angle_dev_ll <= theta_diff <= angle_dev_ul
                        if angle_check:
                            breach = True
                        else:
                            breach = True
                            ray_length = -1
                    else:
                        breach = True
                else:
                    breach = True
                    ray_length = -1

            if not breach:
                ray_indices[ray_pointer] = [niy, nix]
                ray_pointer += 1

        ray_length = np.int32(ray_length)
        if ray_length >= min_stroke_width:
            _ray_iy = ray_indices[:ray_pointer, 0]
            _ray_ix = ray_indices[:ray_pointer, 1]
            for each_y, each_x in zip(_ray_iy, _ray_ix):
                sw_val = swt_matrix[each_y, each_x]
                if ray_length > sw_val:
                    swt_matrix[each_y, each_x] = ray_length

    return swt_matrix


try:
    swt_strokes_jitted = nb.njit(cache=True)(swt_strokes)
except RuntimeError as e:
    # HACK : This is specifically to facilitate the building of `readthedocs`
    # TODO : documentations.
    swt_strokes_jitted = nb.njit(cache=False)(swt_strokes)
except:
    raise


# mask_arr = np.full(shape=(100, 100), fill_value=0, dtype=np.uint8)
# proxyletters_spec = [('label', nb.typeof(99999)),
#                      ('sw_median', nb.typeof(999.999)),
#                      ('color_median', nb.typeof(999.999)),
#                      ('min_height', nb.typeof(999.999)),
#                      ('min_angle', nb.typeof(999.999)),
#                      ('inflated_radius', nb.typeof(999.999)),
#                      ('circular_mask', nb.typeof(mask_arr)),
#                      ('min_label_mask', nb.typeof(mask_arr))]


# @nb.experimental.jitclass(spec=proxyletters_spec)
class ProxyLetter:
    """
    A proxy class for the ``Letters`` object, housing only those properties which
    would be required by the Fusion Class. This is to support application of `numba`
    onto the Fusion Class as the ``Letter`` class object wont be acceptable by Fusion class
    were it to be run on nopython-jit mode
    """

    def __init__(self,
                 label,
                 sw_median,
                 color_median,
                 min_height,
                 min_angle,
                 inflated_radius,
                 circular_mask,
                 min_label_mask):
        """
        Create a ProxyLetter object
        Args:
            label (int) : Letter identifier

            sw_median (float) : Median stroke width of this letter

            color_median (float) : Median Color of this letter

            min_height (int) : Minimum Bounding Box height of this letter

            min_angle (float) : Rotation angle of the Minimum Bounding Box of this letter

            inflated_radius (int) : Inflated Circum-Radius of the Minimum Bounding Box of this letter

            circular_mask (np.ndarray) : Circular filled mask of this letter of radius=inflated_radius and

            centre=Centre Co-Ordinates of the Minimum Bounding Box.

            min_label_mask (np.ndarray) : Filled Minimum Bounding Box of the letter
        """
        # Initialisations
        self.label = label
        self.sw_median = sw_median
        self.color_median = color_median
        self.min_height = min_height
        self.min_angle = min_angle
        self.inflated_radius = inflated_radius
        self.circular_mask = circular_mask
        self.min_label_mask = min_label_mask


# @nb.experimental.jitclass(spec=)
class Fusion:
    """
    Class for fusing Individual Components (Letters) into Grouped Components Words,
    comparing aspects like :
        - Proximity of letters to each other
        - Relative minimum bounding box rotation angle from each other
        - Deviation in color between from one component to the other
        - Ratio of stroke widths from one to the other
        - Ratio of minimum bounding box height of one to the other
    """

    def __init__(self, letters: dict,
                 acceptable_stroke_width_ratio: float,
                 acceptable_color_deviation: List[int],
                 acceptable_height_ratio: float,
                 acceptable_angle_deviation: float):
        """
        Create ``Fusion`` object

        Args:
            letters (List[ProxyLetter]) : List of all the letters to be considered in the fusion pool.

            acceptable_stroke_width_ratio (float) : When comparing two individual components, maximum
             stroke width ratio between two individual components beyond which the component wont be fused together.

            acceptable_color_deviation (List[int]) : When comparing two individual components, maximum color
             deviation between two individual components beyond which the components wont be fused together.

            acceptable_height_ratio (float) : When comparing two individual components, maximum height
             ratio between two individual components beyond which the components wont be fused together.

            acceptable_angle_deviation (float) : When comparing two individual components, maximum angle
             (Minimum Bounding Box Rotation Angle) deviation between two individual components
              beyond which the components wont be fused together.
        """
        self.all_letters: dict = letters
        self.all_words = []
        self.sw_ul = acceptable_stroke_width_ratio
        self.sw_ll = 1 / self.sw_ul
        self.cd_ul = np.linalg.norm(acceptable_color_deviation)
        self.cd_ll = 1 / self.cd_ul
        self.ht_ul = acceptable_height_ratio
        self.ht_ll = 1 / self.ht_ul
        self.ad = np.deg2rad(acceptable_angle_deviation)
        self.letter_masks = [letter.min_label_mask for _, letter in self.all_letters.items()]
        self.letter_masks = np.dstack(self.letter_masks)

    def getProximityLetters(self, anchor_letter: ProxyLetter, remaining_letters: dict) -> List[int]:
        """
        Finds all the labels which are in proximity of anchor_letter amongst the remaining_letters

        Args:
            anchor_letter (ProxyLetter) : Letter with respect to which proximity labels are
             to be searched.

            remaining_letters (dict) : A dictionary, with labels as keys, mapped to their
             corresponding ProxyLetter object.
        Returns:
            (List[int]) :  List of all the labels which are in the proximity of anchor_letter
        """
        idx_y, idx_x = anchor_letter.circular_mask.nonzero()
        proximity_letter_labels = np.unique(self.letter_masks[idx_y, idx_x, :])
        remaining_labels = set(remaining_letters.keys())
        proximity_letter_labels = list(remaining_labels.intersection(proximity_letter_labels))
        return proximity_letter_labels

    def groupEligibility(self, curr_letter, proximity_letter) -> bool:
        """
        Check whether two ProxyLetters are eligible to be grouped with one another.

        Args:
              curr_letter (ProxyLetter) : Current Letter
              proximity_letter (ProxyLetter) : A letter in proximity of Current Letter
        Returns:
            (bool) : Whether curr_letter and proximity_letter are eligible to be grouped with each other
        """
        cl: ProxyLetter = curr_letter
        pl: ProxyLetter = proximity_letter
        # Is there much difference between the font colors of curr_letter proximity_letter
        sw_check = self.sw_ll <= cl.sw_median / pl.sw_median <= self.sw_ul
        # Is there much difference between the median stroke widths of curr_letter proximity_letter
        cd_check = self.cd_ll <= cl.color_median / pl.color_median <= self.cd_ul
        # Is there much difference between the heights of curr_letter proximity_letter
        ht_check = self.ht_ll <= cl.min_height / pl.min_height <= self.ht_ul
        # Is there much difference between the inclination of curr_letter proximity_letter
        ad_check = abs(cl.min_angle - pl.min_angle) <= self.ad

        return sw_check and cd_check and ht_check and ad_check

    def groupLetters(self, curr_letter, remaining_letters, grouping) -> List[ProxyLetter]:
        """
        Groups curr_letter with its proximity labels which are eligible to be grouped
        to it.
        [Recursive Function]

        Args:
            curr_letter (ProxyLetter) : ``ProxyLetter`` whose grouping needs to be mapped.

            remaining_letters (dict) : Dictionary with keys as the ProxyLetter label and the
             corresponding values as the ProxyLetter themselves.

            grouping (list) : A list of lists containing ProxyLetters which can be
             assumed to be *words*.
        Returns:
            (List[ProxyLetter]) : A list containing ProxyLetters which can be
             assumed to be a *word*.
        """
        proximity_letter_labels = self.getProximityLetters(anchor_letter=curr_letter,
                                                           remaining_letters=remaining_letters)
        if not proximity_letter_labels:
            return grouping

        for each_proximity_letter_label in proximity_letter_labels:
            proximity_letter = remaining_letters.get(each_proximity_letter_label)
            # NOTE : Since it is possible that at lower depth calls,
            # some of `each_proximity_letter_label`might have been consumed. That's why
            if proximity_letter:
                if self.groupEligibility(curr_letter=curr_letter, proximity_letter=proximity_letter):
                    confirmed_proximity_letter = remaining_letters.pop(each_proximity_letter_label)
                    grouping.append(confirmed_proximity_letter)
                    grouping = self.groupLetters(curr_letter=confirmed_proximity_letter,
                                                 remaining_letters=remaining_letters,
                                                 grouping=grouping)
        return grouping

    def runGrouping(self) -> List[List[ProxyLetter]]:
        """
        Fuses eligible individual components (letters) together which can be eligible to form
        a *word* out of them.

        Returns:
            (List[List[ProxyLetter]]) : A list of lists containing ProxyLetter which can be assumed
             to be *words* amongst the pool of individual components provided to the ``Fusion`` class.
        """
        while self.all_letters:
            # Get the next letter to make a group from
            next_letter_label, next_letter = self.all_letters.popitem()
            # Get all the letters which can belong to `next_letter`
            letter_group = self.groupLetters(curr_letter=next_letter,
                                             remaining_letters=self.all_letters,
                                             grouping=[next_letter])

            # Append this word to all_words
            self.all_words.append(letter_group)

        return self.all_words
