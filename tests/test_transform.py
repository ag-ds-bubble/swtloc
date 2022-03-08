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


class TestTransformMethods(unittest.TestCase):
    tdata = None
    swtl = None

    @classmethod
    def setUpClass(cls) -> None:
        print("\n\n=============================TRANSFORM_IMAGE================================")
        print('Loading the testing data ...')
        with open(TEST_DATA_PATH, 'rb') as handle:
            cls.tdata = pickle.load(handle)
        print('Prepping & Run Once the `swtl` object for facilitating numba jit')
        cls.swtl = SWTLocalizer(images=[cls.tdata.get('swtimage.transformimage').get('inp_img1'),
                                        cls.tdata.get('swtimage.transformimage').get('inp_img2')])
        cls.swtImgObj0 = cls.swtl.swtimages[0]
        cls.swtImgObj1 = cls.swtl.swtimages[1]
        _ = cls.swtl.swtimages[0].transformImage(display=False)

    def setUp(self):
        self.ts = time.perf_counter()
        print(f'\nRunning Test for : swtimage.transformImage.{self.shortDescription()}... ', end='')

    def test_text_mode(self):
        """text_mode"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_tm.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_tm.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='lb_df', display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_engine_time(self):
        """engine_time"""
        t1, t2 = [], []
        for _ in range(4):
            _t1 = self.swtImgObj0.transformImage(text_mode='db_lf', display=False)
            t1.append(float(self.swtImgObj0.transform_time.split()[0]))
            _t2 = self.swtImgObj0.transformImage(text_mode='db_lf', engine='python', display=False)
            t2.append(float(self.swtImgObj0.transform_time.split()[0]))
        eff_inc = np.mean(t2) / np.mean(t1)
        self.assertTrue(eff_inc > 15.0, msg=f'The calculated engine efficiency {eff_inc}')

    def test_gaussian_blurr(self):
        """gaussian_blurr"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_gb.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_gb.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', gaussian_blurr=True, display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='db_lf', gaussian_blurr=False, display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_gaussian_blurr_kernel(self):
        """gaussian_blurr_kernel"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_gbk.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_gbk.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', gaussian_blurr=True,
                                              gaussian_blurr_kernel=(7, 7), display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='db_lf', gaussian_blurr=True,
                                              gaussian_blurr_kernel=(13, 13), display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_edge_function(self):
        """edge_function"""

        def custom_edge_func(gray_image):
            gauss_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
            laplacian_conv = cv2.Laplacian(gauss_image, -1, (5, 5))
            canny_edge = cv2.Canny(laplacian_conv, 20, 140)
            return canny_edge

        out_img1 = self.tdata.get('swtimage.transformimage').get('param_efce.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_efce.mat2')
        res1 = self.swtImgObj1.transformImage(edge_function=custom_edge_func, display=False)
        res2 = self.swtImgObj1.transformImage(edge_function='ac', display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_auto_canny_sigma(self):
        """auto_canny_sigma"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_eac_sig.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_eac_sig.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', auto_canny_sigma=0.1, display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='db_lf', auto_canny_sigma=1.0, display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_include_edges_in_swt(self):
        """include_edges_in_swt"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_ince.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_ince.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', include_edges_in_swt=True,
                                              gaussian_blurr_kernel=(11, 11), display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='db_lf', include_edges_in_swt=False,
                                              gaussian_blurr_kernel=(11, 11), display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_minimum_stroke_width(self):
        """minimum_stroke_width"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_misw.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_misw.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', minimum_stroke_width=10,
                                              include_edges_in_swt=False, display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='db_lf', minimum_stroke_width=20,
                                              include_edges_in_swt=False, display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_maximum_stroke_width(self):
        """maximum_stroke_width"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_masw.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_masw.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', maximum_stroke_width=20, display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='db_lf', maximum_stroke_width=70, display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_check_angle_deviation(self):
        """check_angle_deviation"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_cad.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_cad.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', check_angle_deviation=True, display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='db_lf', check_angle_deviation=False, display=False)
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def test_maximum_angle_deviation(self):
        """maximum_angle_deviation"""
        out_img1 = self.tdata.get('swtimage.transformimage').get('param_mad.mat1')
        out_img2 = self.tdata.get('swtimage.transformimage').get('param_mad.mat2')
        res1 = self.swtImgObj0.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi / 8, display=False)
        res2 = self.swtImgObj0.transformImage(text_mode='db_lf', maximum_angle_deviation=np.pi / 2, display=False)
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
