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


class TestLettersMethods(unittest.TestCase):
    swtImgObj0 = None
    swtImgObj1 = None
    swtl = None
    tdata = None

    @classmethod
    def setUpClass(cls) -> None:
        print("\n\n=============================LOCALIZE_LETTERS================================")
        print('Loading the testing data ...')
        with open(TEST_DATA_PATH, 'rb') as handle:
            cls.tdata = pickle.load(handle)
        print('Prepping & Run Once the `swtl` object for facilitating numba jit')
        cls.swtl = SWTLocalizer(images=[cls.tdata.get('swtimage.letters').get('inp_img1'),
                                        cls.tdata.get('swtimage.letters').get('inp_img2')])
        cls.swtImgObj0 = cls.swtl.swtimages[0]
        cls.swtImgObj1 = cls.swtl.swtimages[1]
        _ = cls.swtImgObj0.transformImage(text_mode='lb_df',
                                          maximum_angle_deviation=np.pi / 4,
                                          include_edges_in_swt=True,
                                          display=False)
        _ = cls.swtImgObj1.transformImage(text_mode='db_lf',
                                          maximum_stroke_width=30,
                                          maximum_angle_deviation=np.pi / 8,
                                          display=False)

    def setUp(self):
        self.ts = time.perf_counter()
        print(f'\nRunning Test for : swtimage.transformImage.{self.shortDescription()}... ', end='')

    def test_minimum_pixels_per_cc(self):
        """minimum_pixels_per_cc"""
        out_img1 = self.tdata.get('swtimage.letters').get('param_mippc.mat1')
        out_img2 = self.tdata.get('swtimage.letters').get('param_mippc.mat2')
        _ = self.swtImgObj0.localizeLetters(display=False)
        res1 = self.swtImgObj0.image_original_masked_letter_localized
        _ = self.swtImgObj0.localizeLetters(minimum_pixels_per_cc=9000, display=False)
        res2 = self.swtImgObj0.image_original_masked_letter_localized
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_maximum_pixels_per_cc(self):
        """maximum_pixels_per_cc"""
        out_img1 = self.tdata.get('swtimage.letters').get('param_mxppc.mat1')
        out_img2 = self.tdata.get('swtimage.letters').get('param_mxppc.mat2')
        _ = self.swtImgObj0.localizeLetters(maximum_pixels_per_cc=3_000, display=False)
        res1 = self.swtImgObj0.image_original_masked_letter_localized
        _ = self.swtImgObj0.localizeLetters(maximum_pixels_per_cc=5_000, display=False)
        res2 = self.swtImgObj0.image_original_masked_letter_localized
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_acceptable_aspect_ratio(self):
        """acceptable_aspect_ratio"""
        out_img1 = self.tdata.get('swtimage.letters').get('param_aar.mat1')
        out_img2 = self.tdata.get('swtimage.letters').get('param_aar.mat2')
        _ = self.swtImgObj1.localizeLetters(acceptable_aspect_ratio=0.5, display=False)
        res1 = self.swtImgObj1.image_original_masked_letter_localized
        _ = self.swtImgObj1.localizeLetters(acceptable_aspect_ratio=0.005, display=False)
        res2 = self.swtImgObj1.image_original_masked_letter_localized
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_localize_by(self):
        """localize_by"""
        out_img1 = self.tdata.get('swtimage.letters').get('param_lb.mat1')
        out_img2 = self.tdata.get('swtimage.letters').get('param_lb.mat2')
        out_img3 = self.tdata.get('swtimage.letters').get('param_lb.mat3')
        _ = self.swtImgObj1.localizeLetters(localize_by='min_bbox', display=False)
        res1 = self.swtImgObj1.image_original_masked_letter_localized
        _ = self.swtImgObj1.localizeLetters(localize_by='ext_bbox', display=False)
        res2 = self.swtImgObj1.image_original_masked_letter_localized
        _ = self.swtImgObj1.localizeLetters(localize_by='outline', display=False)
        res3 = self.swtImgObj1.image_original_masked_letter_localized
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))
        self.assertTrue(np.array_equal(res3, out_img3, equal_nan=True))

    def test_padding_pct(self):
        """padding_pct"""
        out_img1 = self.tdata.get('swtimage.letters').get('param_pp.mat1')
        out_img2 = self.tdata.get('swtimage.letters').get('param_pp.mat2')
        _ = self.swtImgObj1.localizeLetters(display=False)
        res1 = self.swtImgObj1.image_original_masked_letter_localized
        _ = self.swtImgObj1.localizeLetters(padding_pct=0.1, display=False)
        res2 = self.swtImgObj1.image_original_masked_letter_localized
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def tearDown(self) -> None:
        time_taken = round(time.perf_counter() - self.ts, 3)
        print(f'[Time Taken : {time_taken} sec]')

    @classmethod
    def tearDownClass(cls) -> None:
        print("\n======================================================================")


if __name__ == '__main__':
    unittest.main()
