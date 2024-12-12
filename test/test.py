import unittest
import sys
import os
import numpy as np
import scipy.interpolate

sys.path.append('../src/stonp/')
from stacker import Stacker
cwd = os.getcwd() + '/'

class TestJsonLoader(unittest.TestCase):
    def test_no_args(self):
        with self.assertRaises(TypeError):
            Stacker._json_loader()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Stacker._json_loader('test.json')

    def test_bad_file_argument(self):
        with self.assertRaises(TypeError):
            Stacker._json_loader(False)
            Stacker._json_loader(1)
            Stacker._json_loader(1.23)
            Stacker._json_loader([])

    def test_bad_df_arg(self):
        with self.assertRaises(TypeError):
            Stacker._json_loader(cwd+'../filters/test_bands.json', df=True)
            Stacker._json_loader(cwd+'../filters/test_bands.json', df='a')
            Stacker._json_loader(cwd+'../filters/test_bands.json', df=7)
            Stacker._json_loader(cwd+'../filters/test_bands.json', df=1.23)
            Stacker._json_loader(cwd+'../filters/test_bands.json', df=[])

    def test_bad_sort_arg(self):
        with self.assertRaises(TypeError):
            Stacker._json_loader(cwd+'../filters/test_bands.json', sort='a')
            Stacker._json_loader(cwd+'../filters/test_bands.json', sort=1)
            Stacker._json_loader(cwd+'../filters/test_bands.json', sort=1.2)
            Stacker._json_loader(cwd+'../filters/test_bands.json', sort=[])

    def test_returns_len(self):
        self.assertEqual(len(Stacker._json_loader(
            cwd+'../filters/test_bands.json')), 4)

    def test_returns_type(self):
        nb_labels, wl_nb, r_nb, wl_grid_obs = Stacker._json_loader(
            cwd+'../filters/test_bands.json')
        self.assertIsInstance(nb_labels, list)
        self.assertIsInstance(wl_nb, np.ndarray)
        self.assertIsInstance(r_nb, scipy.interpolate.interp1d)
        self.assertIsInstance(wl_grid_obs, np.ndarray)

if __name__ == "__main__":
    unittest.main()
