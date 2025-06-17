import unittest
from unittest.mock import patch
from io import StringIO
import os
import numpy as np
import scipy.interpolate
import stonp

cwd = os.getcwd() + '/'
repo_home = cwd + '../'


class TestLinterp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def test_wrong_arg_number(self):
        with self.assertRaises(TypeError):
            stonp.stacker.linterp()
            stonp.stacker.linterp(1)
            stonp.stacker.linterp(1, 2)
            stonp.stacker.linterp(1, 2, 3)
            stonp.stacker.linterp(1, 2, 3, 4, 5)

    def test_bad_first_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.linterp(False, [], [], [])
            stonp.stacker.linterp(1, [], [], [])
            stonp.stacker.linterp(1.23, [], [], [])
            stonp.stacker.linterp('a', [], [], [])
            stonp.stacker.linterp(object(), [], [], [])

    def test_bad_second_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.linterp([], False, [], [])
            stonp.stacker.linterp([], 1, [], [])
            stonp.stacker.linterp([], 1.23, [], [])
            stonp.stacker.linterp([], 'a', [], [])
            stonp.stacker.linterp([], object(), [], [])

    def test_bad_third_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.linterp([], [], False, [])
            stonp.stacker.linterp([], [], 1, [])
            stonp.stacker.linterp([], [], 1.23, [])
            stonp.stacker.linterp([], [], 'a', [])
            stonp.stacker.linterp([], [], object(), [])

    def test_bad_fourth_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.linterp([], [], [], False)
            stonp.stacker.linterp([], [], [], 1)
            stonp.stacker.linterp([], [], [], 1.23)
            stonp.stacker.linterp([], [], [], 'a')
            stonp.stacker.linterp([], [], [], object())

    def test_bad_arg_sizes(self):
        with self.assertRaises(ValueError):
            stonp.stacker.linterp([12], [], [1], [1])
            stonp.stacker.linterp([12], [1], [], [1])
            stonp.stacker.linterp([12], [1], [1], [])
            stonp.stacker.linterp([12], [1], [1, 2], [])


class TestJsonLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def test_no_args(self):
        with self.assertRaises(TypeError):
            stonp.stacker.json_loader()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            stonp.stacker.json_loader('test.json')

    def test_bad_file_argument(self):
        with self.assertRaises(TypeError):
            stonp.stacker.json_loader(False)
            stonp.stacker.json_loader(1)
            stonp.stacker.json_loader(1.23)
            stonp.stacker.json_loader(object())

    def test_bad_df_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', df=True)
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', df='a')
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', df=7)
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', df=1.23)
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', df=object())

    def test_bad_sort_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', sort='a')
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', sort=1)
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', sort=1.2)
            stonp.stacker.json_loader(
                repo_home+'filters/test_bands.json', sort=object())

    def test_returns_len(self):
        self.assertEqual(
            len(stonp.stacker.json_loader(repo_home+'filters/test_bands.json')), 4)

    def test_returns_type(self):
        nb_labels, wl_nb, r_nb, wl_grid_obs = stonp.stacker.json_loader(
            repo_home+'filters/test_bands.json')
        self.assertIsInstance(nb_labels, list)
        self.assertIsInstance(wl_nb, np.ndarray)
        self.assertIsInstance(r_nb, scipy.interpolate.interp1d)
        self.assertIsInstance(wl_grid_obs, np.ndarray)


class TestBinDictParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def test_no_args(self):
        with self.assertRaises(TypeError):
            stonp.stacker.bin_dict_parser()

    def test_bad_dict_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.bin_dict_parser(False)
            stonp.stacker.bin_dict_parser(1)
            stonp.stacker.bin_dict_parser(1.23)
            stonp.stacker.bin_dict_parser(object())

    def test_returns_type(self):
        self.assertIsInstance(stonp.stacker.bin_dict_parser(
            {'test': [1, 2, 3]}), dict)

    def test_dic_struct(self):
        bins = stonp.stacker.bin_dict_parser(
            {'test1': [1, 2, 3], 'test2': [4, 5, 6]})
        self.assertEqual(len(bins), 2)
        for key in bins:
            for val in bins[key]:
                self.assertIsInstance(val, list)
                self.assertEqual(len(val), 2)

    def test_dict_values(self):
        bins = stonp.stacker.bin_dict_parser(
            {'test1': [1, 2, 3], 'test2': [4, 5, 6]})
        self.assertListEqual(bins['test1'][0], [1, 2])
        self.assertListEqual(bins['test1'][1], [2, 3])
        self.assertListEqual(bins['test2'][0], [4, 5])
        self.assertListEqual(bins['test2'][1], [5, 6])


class TestDetermineColsRows(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def test_bad_arg_number(self):
        with self.assertRaises(TypeError):
            stonp.stacker.determine_cols_rows()
            stonp.stacker.determine_cols_rows(1)
            stonp.stacker.determine_cols_rows(1, 2, 3)

    def test_bad_first_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.determine_cols_rows(None, 1.23)
            stonp.stacker.determine_cols_rows(True, 1.23)
            stonp.stacker.determine_cols_rows('a', 1.23)
            stonp.stacker.determine_cols_rows(object(), 1.23)

    def test_bad_second_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.determine_cols_rows(1, None)
            stonp.stacker.determine_cols_rows(1, True)
            stonp.stacker.determine_cols_rows(1, 'a')
            stonp.stacker.determine_cols_rows(1, object())

    def test_return_type(self):
        self.assertIsInstance(
            stonp.stacker.determine_cols_rows(1, 1), tuple)
        a, b = stonp.stacker.determine_cols_rows(1, 1)
        self.assertIsInstance(a, int)
        self.assertIsInstance(b, int)

    def test_return_size(self):
        self.assertEqual(len(stonp.stacker.determine_cols_rows(1, 1)), 2)

    def test_return_values(self):
        self.assertEqual(stonp.stacker.determine_cols_rows(4, 1), (2, 2))


class TestQueryYesNo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def test_bad_arg_number(self):
        with self.assertRaises(TypeError):
            stonp.stacker.query_yes_no()
            stonp.stacker.query_yes_no("a", "b", "c")

    def test_bad_first_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.query_yes_no(1)
            stonp.stacker.query_yes_no(1.2)
            stonp.stacker.query_yes_no(None)
            stonp.stacker.query_yes_no(object())

    def test_bad_second_arg(self):
        with self.assertRaises(ValueError):
            stonp.stacker.query_yes_no("question", 1)
            stonp.stacker.query_yes_no("question", 1.2)
            stonp.stacker.query_yes_no("question", None)
            stonp.stacker.query_yes_no("question", object())
            stonp.stacker.query_yes_no("question", "maybe")

    @patch("builtins.input", side_effect=["yes", "y", "ye", "YES", "Y", "YE"])
    @patch("sys.stdout")
    def test_positive_answers(self, mock_in, mock_out):
        self.assertTrue(stonp.stacker.query_yes_no("question"))

    @patch("builtins.input", side_effect=["n", "N", "no", "NO"])
    @patch("sys.stdout")
    def test_negative_answers(self, mock_in, mock_out):
        self.assertFalse(stonp.stacker.query_yes_no("question"))

    @patch("builtins.input", return_value="")
    @patch("sys.stdout")
    def test_default_yes(self, mock_in, mock_out):
        self.assertTrue(stonp.stacker.query_yes_no("question", "yes"))

    @patch("builtins.input", return_value="")
    @patch("sys.stdout")
    def test_default_no(self, mock_in, mock_out):
        self.assertFalse(stonp.stacker.query_yes_no("question", "no"))

    @patch("sys.stdout", new_callable=StringIO)
    @patch("builtins.input", side_effect=["", "y"])
    def test_default_answers(self, mock_in, mock_out):
        res = stonp.stacker.query_yes_no("Question", default=None)
        output = mock_out.getvalue()
        self.assertIn(
            "Please respond with 'yes' or 'no' (or 'y' or 'n').\n", output)
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
