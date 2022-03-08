# Five test images to be used as muse
# Replicate from the notebooks

import time
import sys
if any([k in sys.version[:4] for k in ['3.6', '3.7']]):
    import pickle5 as pickle
else:
    import pickle
import unittest

import numpy as np
from cv2 import cv2

from . import TEST_DATA_PATH

from swtloc import SWTLocalizer
from swtloc.utils import SWTLocalizerValueError


class TestSWTLocalizerMethods(unittest.TestCase):
    tdata = None

    @classmethod
    def setUpClass(cls) -> None:
        print("\n\n=============================SWTLOCALIZER================================")
        print('Loading the testing data ...')
        with open(TEST_DATA_PATH, 'rb') as handle:
            cls.tdata = pickle.load(handle)
        cls.ipaths = cls.tdata.get('swtl').get('fpaths')

    def setUp(self):
        self.ts = time.perf_counter()
        print(f'\nRunning Test for : swtlocalizer.{self.shortDescription()}... ', end='')

    def test_no_input(self):
        """no_input"""
        # TODO : Activate from v2.1.0
        # with self.assertRaises(SWTLocalizerValueError) as context:
        #     SWTLocalizer()
        pass

    def test_single_fpath(self):
        """singe_file_path"""
        try:
            SWTLocalizer(image_paths=self.ipaths[0])
        except Exception as e:
            self.fail(f"myFunc() raised {e} unexpectedly!")

    def test_multiple_fpath(self):
        """multiple_file_path"""
        try:
            SWTLocalizer(image_paths=self.ipaths)
        except Exception as e:
            self.fail(f"myFunc() raised {e} unexpectedly!")

    def test_single_image(self):
        """single_image"""
        try:
            SWTLocalizer(images=cv2.imread(self.ipaths[0]))
        except Exception as e:
            self.fail(f"myFunc() raised {e} unexpectedly!")

    def test_multiple_images(self):
        """multiple_images"""
        try:
            SWTLocalizer(images=[cv2.imread(k) for k in self.ipaths])
        except Exception as e:
            self.fail(f"myFunc() raised {e} unexpectedly!")

    def test_mixed_input(self):
        """mixed_input"""
        mix_inp1 = [self.ipaths[0], cv2.imread(self.ipaths[-1])]
        mix_inp2 = [1, True, 'abc']
        with self.assertRaises(SWTLocalizerValueError) as context:
            SWTLocalizer(image_paths=mix_inp1)
        with self.assertRaises(SWTLocalizerValueError) as context:
            SWTLocalizer(image_paths=mix_inp2)
        with self.assertRaises(SWTLocalizerValueError) as context:
            SWTLocalizer(images=mix_inp1)
        with self.assertRaises(SWTLocalizerValueError) as context:
            SWTLocalizer(images=mix_inp2)

    def test_image_dimensions(self):
        """image_dimensions"""
        img_5d = np.full(shape=(5, 5, 5, 5, 5), fill_value=255)
        img_5c = np.full(shape=(5, 5, 5), fill_value=255)

        with self.assertRaises(SWTLocalizerValueError) as context:
            SWTLocalizer(images=img_5c)
        with self.assertRaises(SWTLocalizerValueError) as context:
            SWTLocalizer(images=img_5d)



    def tearDown(self) -> None:
        time_taken = round(time.perf_counter() - self.ts, 3)
        print(f'[Time Taken : {time_taken} sec]')

    @classmethod
    def tearDownClass(cls) -> None:
        print("\n======================================================================")


if __name__ == '__main__':
    unittest.main()
