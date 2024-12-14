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


class TestBinDictParser(unittest.TestCase):
    def test_no_args(self):
        with self.assertRaises(TypeError):
            Stacker._bin_dict_parser()

    def test_bad_dict_arg(self):
        with self.assertRaises(TypeError):
            Stacker._bin_dict_parser(False)
            Stacker._bin_dict_parser(1)
            Stacker._bin_dict_parser(1.23)
            Stacker._bin_dict_parser([])

    def test_returns_type(self):
        self.assertIsInstance(Stacker._bin_dict_parser(
            {'test': [1, 2, 3]}), dict)

    def test_dic_struct(self):
        bins = Stacker._bin_dict_parser(
            {'test1': [1, 2, 3], 'test2': [4, 5, 6]})
        self.assertEqual(len(bins), 2)
        for key in bins:
            for val in bins[key]:
                self.assertIsInstance(val, list)
                self.assertEqual(len(val), 2)

    def test_dict_values(self):
        bins = Stacker._bin_dict_parser(
            {'test1': [1, 2, 3], 'test2': [4, 5, 6]})
        self.assertListEqual(bins['test1'][0], [1, 2])
        self.assertListEqual(bins['test1'][1], [2, 3])
        self.assertListEqual(bins['test2'][0], [4, 5])
        self.assertListEqual(bins['test2'][1], [5, 6])
if __name__ == "__main__":
    unittest.main()
