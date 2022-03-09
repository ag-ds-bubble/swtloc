import time
import sys
if any([k in sys.version[:4] for k in ['3.6', '3.7']]):
    import pickle5 as pickle
else:
    import pickle
import unittest

import numpy as np

from . import TEST_DATA_PATH

from swtloc import SWTLocalizer


class TestWordsMethods(unittest.TestCase):
    swtImgObj0 = None
    swtl = None
    tdata = None

    @classmethod
    def setUpClass(cls) -> None:
        print("\n\n=============================LOCALIZE_WORDS================================")
        print('Loading the testing data ...')
        with open(TEST_DATA_PATH, 'rb') as handle:
            cls.tdata = pickle.load(handle)
        print('Prepping & Run Once the `swtl` object for facilitating numba jit')
        cls.swtl = SWTLocalizer(images=cls.tdata.get('swtimage.words').get('inp_img1'))
        cls.swtImgObj0 = cls.swtl.swtimages[0]
        _ = cls.swtImgObj0.transformImage(auto_canny_sigma=1.0,
                                          minimum_stroke_width=3,
                                          maximum_stroke_width=20,
                                          maximum_angle_deviation=np.pi / 6,
                                          display=False)
        _ = cls.swtImgObj0.localizeLetters(display=False)

    def setUp(self):
        self.ts = time.perf_counter()
        print(f'\nRunning Test for : swtimage.transformImage.{self.shortDescription()}... ', end='')

    def test_localize_by(self):
        """localize_by"""
        out_img1 = self.tdata.get('swtimage.words').get('param_lb.mat1')
        out_img2 = self.tdata.get('swtimage.words').get('param_lb.mat2')
        out_img3 = self.tdata.get('swtimage.words').get('param_lb.mat3')
        _ = self.swtImgObj0.localizeWords(localize_by='bubble', display=False)
        res1 = self.swtImgObj0.image_original_masked_word_localized
        _ = self.swtImgObj0.localizeWords(localize_by='bbox', display=False)
        res2 = self.swtImgObj0.image_original_masked_word_localized
        _ = self.swtImgObj0.localizeWords(localize_by='polygon', display=False)
        res3 = self.swtImgObj0.image_original_masked_word_localized
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))
        self.assertTrue(np.array_equal(res3, out_img3, equal_nan=True))

    def tearDown(self) -> None:
        time_taken = round(time.perf_counter() - self.ts, 3)
        print(f'[Time Taken : {time_taken} sec]')

    @classmethod
    def tearDownClass(cls) -> None:
        print("\n======================================================================")


if __name__ == '__main__':
    unittest.main()
