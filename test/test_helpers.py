import unittest
import os
import sys

from helpers import create_mockfile, skip_if_frequency_deactivated, skip_if_wavelength_deactivated, skip_if_slow_deactivated


class TestGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def test_no_args(self):
        with self.assertRaises(TypeError):
            create_mockfile()

    def test_only_first_arg(self):
        with self.assertRaises(TypeError):
            create_mockfile(spectral_density='wavelength')

    def test_only_second_arg(self):
        with self.assertRaises(TypeError):
            create_mockfile(constant_luminosity=True)

    def test_bad_first_arg(self):
        with self.assertRaises(ValueError):
            create_mockfile(spectral_density='bad_value',
                            constant_luminosity=True)

        with self.assertRaises(TypeError):
            create_mockfile(spectral_density=1,
                            constant_luminosity=True)
            create_mockfile(spectral_density=True,
                            constant_luminosity=True)
            create_mockfile(spectral_density=1.23,
                            constant_luminosity=True)
            create_mockfile(spectral_density=object(),
                            constant_luminosity=True)

    def test_bad_second_arg(self):
        with self.assertRaises(TypeError):
            create_mockfile(spectral_density='wavelength',
                            constant_luminosity='a')
            create_mockfile(spectral_density='wavelength',
                            constant_luminosity=1)
            create_mockfile(spectral_density='wavelength',
                            constant_luminosity=1.23)
            create_mockfile(spectral_density='wavelength',
                            constant_luminosity=object())

    @skip_if_slow_deactivated()
    def test_good_args(self):
        try:
            create_mockfile(spectral_density='wavelength',
                            constant_luminosity=True)
            self.assertTrue(os.path.exists(
                'mock_catalog_test_wavelength_density_constant_luminosity.csv'))
            os.remove(
                'mock_catalog_test_wavelength_density_constant_luminosity.csv')
        except Exception as e:
            assert False, f"Exception raised: {e}"

        try:
            create_mockfile(spectral_density='wavelength',
                            constant_luminosity=False)
            self.assertTrue(os.path.exists(
                'mock_catalog_test_wavelength_density_evolving_luminosity.csv'))
            os.remove(
                'mock_catalog_test_wavelength_density_evolving_luminosity.csv')
        except Exception as e:
            assert False, f"Exception raised: {e}"

        try:
            create_mockfile(spectral_density='frequency',
                            constant_luminosity=True)
            self.assertTrue(os.path.exists(
                'mock_catalog_test_frequency_density_constant_luminosity.csv'))
            os.remove(
                'mock_catalog_test_frequency_density_constant_luminosity.csv')
        except Exception as e:
            assert False, f"Exception raised: {e}"

        try:
            create_mockfile(spectral_density='frequency',
                            constant_luminosity=False)
            self.assertTrue(os.path.exists(
                'mock_catalog_test_frequency_density_evolving_luminosity.csv'))
            os.remove(
                'mock_catalog_test_frequency_density_evolving_luminosity.csv')
        except Exception as e:
            assert False, f"Exception raised: {e}"


class TestSkipper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def test_args_slow(self):
        """
        Test that the function to disable slow tests fails if arguments are received
        """
        with self.assertRaises(TypeError):
            skip_if_slow_deactivated("a")
            skip_if_slow_deactivated(1)
            skip_if_slow_deactivated(1.1)
            skip_if_slow_deactivated(object())
            skip_if_slow_deactivated(False)

    def test_args_frequency(self):
        """
        Test that the function to disable frequency tests fails if arguments are received
        """
        with self.assertRaises(TypeError):
            skip_if_frequency_deactivated("a")
            skip_if_frequency_deactivated(1)
            skip_if_frequency_deactivated(1.1)
            skip_if_frequency_deactivated(object())
            skip_if_frequency_deactivated(False)

    def test_args_wavelength(self):
        """
        Test that the function to disable wavelength tests fails if arguments are received
        """
        with self.assertRaises(TypeError):
            skip_if_wavelength_deactivated("a")
            skip_if_wavelength_deactivated(1)
            skip_if_wavelength_deactivated(1.1)
            skip_if_wavelength_deactivated(object())
            skip_if_wavelength_deactivated(False)

