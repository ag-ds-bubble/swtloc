import time
import pickle
import unittest

from . import TEST_DATA_PATH

import numpy as np

from swtloc import SWTLocalizer
from swtloc.utils import auto_canny
from swtloc.utils import image_1C_to_3C
from swtloc.utils import generate_random_swtimage_names
from swtloc.utils import get_connected_components_with_stats


class TestUtilsMethods(unittest.TestCase):

    tdata = None

    @classmethod
    def setUpClass(cls) -> None:
        print("\n\n==================================UTILS================================")
        print('Loading the testing data ...')
        with open(TEST_DATA_PATH, 'rb') as handle:
            cls.tdata = pickle.load(handle)
        print('Prepping & Run Once the `swtl` object for facilitating numba jit')
        cls.swtl = SWTLocalizer(images=cls.tdata.get('utils.ac').get('inp_img'))
        _ = cls.swtl.swtimages[0].transformImage(display=False)

    def setUp(self):
        self.ts = time.perf_counter()
        print(f'\nRunning Test for : utils.{self.shortDescription()}... ', end='')

    def test_generate_random_swtimage_names(self):
        """generate_random_swtimage_names"""
        res = generate_random_swtimage_names(3)
        self.assertEqual(len(res), 3, )
        self.assertIsInstance(res, list)

    def test_auto_canny(self):
        """auto_canny"""
        # Will match the result of test_img2 with (9, 9)
        # gaussian blurr for sigma=0.33(default)
        inp_img = self.tdata.get('utils.ac').get('inp_img')
        out_img = self.tdata.get('utils.ac').get('out_img')
        res = auto_canny(inp_img)
        self.assertTrue(np.array_equal(res, out_img, equal_nan=True))

    def test_image_1C_to_3C(self):
        """image_1C_to_3C"""
        swtl = SWTLocalizer(images=self.tdata.get('utils.i1c3c').get('inp_img').copy())
        swtl_img_obj = swtl.swtimages[0]
        swtmat = swtl_img_obj.transformImage(text_mode='lb_df', gaussian_blurr_kernel=(11, 11),
                                             include_edges_in_swt=False, display=False)
        swtmat[swtmat.nonzero()] = 255
        _, swtmat, _, _ = get_connected_components_with_stats(swtmat)

        res1 = image_1C_to_3C(swtmat)
        out_img1 = self.tdata.get('utils.i1c3c').get('out_img.default').copy()
        self.assertTrue(np.array_equal(res1, out_img1, equal_nan=True))

        res2 = image_1C_to_3C(swtmat, scale_with_values=True)
        out_img2 = self.tdata.get('utils.i1c3c').get('out_img.scale').copy()
        self.assertTrue(np.array_equal(res2, out_img2, equal_nan=True))

    def tearDown(self) -> None:
        time_taken = round(time.perf_counter() - self.ts, 3)
        print(f'[Time Taken : {time_taken} sec]')

    @classmethod
    def tearDownClass(cls) -> None:
        print("\n======================================================================")


if __name__ == '__main__':
    unittest.main()
