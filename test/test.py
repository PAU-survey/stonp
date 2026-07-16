import stonp
import unittest
import sys
import os
import numpy as np
import astropy.units as u

from helpers import create_mockfile, calculate_md5, skip_if_frequency_deactivated, skip_if_wavelength_deactivated, skip_if_slow_deactivated

cwd = os.getcwd() + '/'
repo_home = cwd


class TestLoadCatalog(unittest.TestCase):
    mock_filename = None

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))
        cls.run_slow_tests = os.getenv("SKIP_SLOW_TESTS") != "1"
        if cls.run_slow_tests:
            create_mockfile(spectral_density='wavelength',
                            constant_luminosity=True)
        cls.mock_filename = 'mock_catalog_test_wavelength_density_constant_luminosity.csv'

    @classmethod
    def tearDownClass(cls):
        if cls.run_slow_tests:
            os.remove(cls.mock_filename)
            cls.mock_filename = None

    def setUp(self):
        self.st = stonp.Stacker()

    def tearDown(self):
        self.st = None

    def test_no_args(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog()

    def test_file_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            self.st.load_catalog('not_existing.csv')

    def test_bad_file_arg(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog(False)
            self.st.load_catalog(1)
            self.st.load_catalog(1.23)
            self.st.load_catalog(object())

    def test_bad_max_nan_bands(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog(cwd + self.mock_filename, max_nan_bands=False)
            self.st.load_catalog(cwd + self.mock_filename, max_nan_bands='a')
            self.st.load_catalog(cwd + self.mock_filename, max_nan_bands=1.23)
            self.st.load_catalog(cwd + self.mock_filename,
                                 max_nan_bands=object())
        with self.assertRaises(ValueError):
            self.st.load_catalog(cwd + self.mock_filename, max_nan_bands=-1)

    def test_bad_z_label(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog(cwd + self.mock_filename, z_label=False)
            self.st.load_catalog(cwd + self.mock_filename, z_label=1)
            self.st.load_catalog(cwd + self.mock_filename, z_label=1.23)
            self.st.load_catalog(cwd + self.mock_filename, z_label=object())
        with self.assertRaises(ValueError):
            self.st.load_catalog(cwd + self.mock_filename, z_label='')
            self.st.load_catalog(cwd + self.mock_filename, z_label='bad_value')

    def test_bad_fill_nans(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog(cwd + self.mock_filename, fill_nans=False)
            self.st.load_catalog(cwd + self.mock_filename, fill_nans=1)
            self.st.load_catalog(cwd + self.mock_filename, fill_nans=1.23)
            self.st.load_catalog(cwd + self.mock_filename, fill_nans=object())
        with self.assertRaises(ValueError):
            self.st.load_catalog(cwd + self.mock_filename,
                                 fill_nans='bad_value')

    def test_bad_band_data(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog(cwd + self.mock_filename, bands_data=False)
            self.st.load_catalog(cwd + self.mock_filename, bands_data=1)
            self.st.load_catalog(cwd + self.mock_filename, bands_data=1.23)
            self.st.load_catalog(cwd + self.mock_filename, bands_data=object())
        with self.assertRaises(FileNotFoundError):
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_data='test.json')
        with self.assertRaises(ValueError):
            self.st.load_catalog(cwd + self.mock_filename, bands_data={})
            self.st.load_catalog(cwd + self.mock_filename, bands_data="")

    @skip_if_slow_deactivated()
    def test_bad_band_keys(self):
        with self.assertRaises(KeyError):
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_data={'NB455': None})
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_data={'NB455': 'a'})
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_data={'NB455': object()})
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_data={'NB455': False})

    def test_bad_bands_error_suffix(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_error_suffix=False)
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_error_suffix=1)
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_error_suffix=1.23)
            self.st.load_catalog(cwd + self.mock_filename,
                                 bands_error_suffix=object())

    def test_bad_flux_units(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog(cwd + self.mock_filename, flux_units=False)
            self.st.load_catalog(cwd + self.mock_filename, flux_units=1)
            self.st.load_catalog(cwd + self.mock_filename, flux_units=1.23)
            self.st.load_catalog(cwd + self.mock_filename, flux_units=object())
        with self.assertRaises(ValueError):
            self.st.load_catalog(cwd + self.mock_filename,
                                 flux_units='bad_value')
            self.st.load_catalog(cwd + self.mock_filename,
                                 flux_units=u.Unit("bar"))

    def test_bad_wavelength_units(self):
        with self.assertRaises(TypeError):
            self.st.load_catalog(cwd + self.mock_filename,
                                 wavelength_units=False)
            self.st.load_catalog(cwd + self.mock_filename, wavelength_units=1)
            self.st.load_catalog(cwd + self.mock_filename,
                                 wavelength_units=1.23)
            self.st.load_catalog(cwd + self.mock_filename,
                                 wavelength_units=object())
        with self.assertRaises(ValueError):
            self.st.load_catalog(cwd + self.mock_filename,
                                 wavelength_units='bad_value')
            self.st.load_catalog(cwd + self.mock_filename,
                                 wavelength_units=u.Unit("bar"))

    @skip_if_slow_deactivated()
    def test_check_all_correct_args(self):
        try:
            self.st.load_catalog(cwd + self.mock_filename, max_nan_bands=0, z_label='z',
                                 fill_nans='zero', bands_data=repo_home + 'filters/test_bands.json',
                                 bands_error_suffix='_error', flux_units=u.Unit('erg / (nm s cm2)'),
                                 wavelength_units='nm')
        except Exception as e:
            assert False, f"Exception raised: {e}"


class TestStonp(unittest.TestCase):
    st = None
    template_numbers = None

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def setUp(self):
        self.st = stonp.Stacker()
        template_numbers, *_ = stonp.stacker.json_loader(
            repo_home+'spectra/blanton2003_sed_templates.json', sort=False)
        self.template_numbers = [int(template_number)
                                 for template_number in template_numbers]
        self.sd = ""
        self.cl = True

    def tearDown(self):
        self.st = None
        self.template_numbers = None
        stack_dirname = 'stack_test_' + self.sd + '_density_'
        if self.cl:
            stack_dirname += 'luminosity'
        else:
            stack_dirname += 'normalized'
        try:
            for fname in sorted(os.listdir(os.path.join(cwd, stack_dirname))):
                os.remove(os.path.join(cwd, stack_dirname, fname))
            os.rmdir(stack_dirname)
        except FileNotFoundError:
            pass

    @skip_if_wavelength_deactivated()
    def test_wavelength_constant_luminosity(self):
        self.sd = 'wavelength'
        self.cl = True
        create_mockfile(spectral_density=self.sd, constant_luminosity=self.cl)
        mock_filename = 'mock_catalog_test_wavelength_density_constant_luminosity.csv'
        self.assertTrue(os.path.exists(mock_filename))
        stack_dirname = 'stack_test_wavelength_density_luminosity'
        flux_units = 'erg / (s cm2 nm)'

        sys.stdout = None
        self.st.load_catalog(mock_filename, bands_data=repo_home +
                             'filters/test_bands.json', z_label='z', flux_units=flux_units)
        sys.stdout = sys.__stdout__
        self.st.to_rest_frame(flux_conversion='luminosity',
                              use_band_responses=True)
        self.st.stack(bin_dict={'template_number==': self.template_numbers},
                      weight='snr_square', error_type='flux_error')
        self.st.stack(
            bin_dict={'template_number==': self.template_numbers}, error_type='std')
        self.st.save_stack(stack_dirname, overwrite=True)
        self.st.plot(line_label='template_number', logscale=True,
                     wavelength_min=200, wavelength_max=700, show=False)
        self.st.plot(column_label='template_number',
                     counts=True, aspect_ratio=2, show=False)
        self.st.plot(row_label='template_number', spectral_lines=True,
                     fig_title=r'test $\alpha$', show=False)
        self.st.load_stack(stack_dirname)
        smoothing_bands = self.st.return_smoothing_bands()
        stack = self.st.return_stack()

        self.assertEqual(np.isnan(stack.data).sum(), 0)
        # test shape
        self.assertGreaterEqual(np.prod(stack.data[:, 1, :]), 0)
        self.assertEqual(stack.data.shape[0], len(self.template_numbers))
        norms = np.trapezoid(stack.data[:, 0, :], stack.rf_wl.data, axis=-1)
        self.assertGreater(np.prod(norms), 0)

        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'smoothing_bands.nc')), '7b6de74a83dc2767fde72e462ae1e4c5')

        os.remove(mock_filename)

    @skip_if_wavelength_deactivated()
    def test_wavelength_evolving_luminosity(self):
        self.sd = 'wavelength'
        self.cl = False
        create_mockfile(spectral_density=self.sd, constant_luminosity=self.cl)
        mock_filename = 'mock_catalog_test_wavelength_density_evolving_luminosity.csv'
        self.assertTrue(os.path.exists(mock_filename))
        stack_dirname = 'stack_test_wavelength_density_normalized'
        flux_units = 'erg / (s cm2 nm)'

        sys.stdout = None
        self.st.load_catalog(mock_filename, bands_data=repo_home +
                             'filters/test_bands.json', z_label='z', flux_units=flux_units)
        sys.stdout = sys.__stdout__
        self.st.to_rest_frame(flux_conversion='normalized',
                              use_band_responses=True)
        self.st.stack(bin_dict={'template_number==': self.template_numbers},
                      weight='snr_square', error_type='flux_error')
        self.st.stack(
            bin_dict={'template_number==': self.template_numbers}, error_type='std')

        self.st.save_stack(stack_dirname, overwrite=True)
        self.st.plot(line_label='template_number', logscale=True,
                     wavelength_min=200, wavelength_max=700, show=False)
        self.st.plot(column_label='template_number',
                     counts=True, aspect_ratio=2, show=False)
        self.st.plot(row_label='template_number', spectral_lines=True,
                     fig_title=r'test $\alpha$', show=False)
        self.st.load_stack(stack_dirname)
        smoothing_bands = self.st.return_smoothing_bands()
        stack = self.st.return_stack()

        self.assertEqual(np.isnan(stack.data).sum(), 0)
        # test shape
        self.assertGreaterEqual(np.prod(stack.data[:, 1, :]), 0)
        self.assertEqual(stack.data.shape[0], len(self.template_numbers))
        norms = np.trapezoid(stack.data[:, 0, :], stack.rf_wl.data, axis=-1)
        wl_span = stack.rf_wl.data[-1] - stack.rf_wl.data[0]
        self.assertEqual(np.prod(np.isclose(norms, wl_span)), 1)

        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'smoothing_bands.nc')), '8033871b7019cf957ac089f77773504f')

        os.remove(mock_filename)

    @skip_if_frequency_deactivated()
    def test_frequency_constant_luminosity(self):
        self.sd = 'frequency'
        self.cl = True
        create_mockfile(spectral_density=self.sd, constant_luminosity=self.cl)
        mock_filename = 'mock_catalog_test_frequency_density_constant_luminosity.csv'
        self.assertTrue(os.path.exists(mock_filename))
        stack_dirname = 'stack_test_frequency_density_luminosity'
        flux_units = 'erg / (s cm2 Hz)'

        sys.stdout = None
        self.st.load_catalog(mock_filename, bands_data=repo_home +
                             'filters/test_bands.json', z_label='z', flux_units=flux_units)
        sys.stdout = sys.__stdout__
        self.st.to_rest_frame(flux_conversion='luminosity',
                              use_band_responses=True)
        self.st.stack(bin_dict={'template_number==': self.template_numbers},
                      weight='snr_square', error_type='flux_error')
        self.st.stack(
            bin_dict={'template_number==': self.template_numbers}, error_type='std')

        self.st.save_stack(stack_dirname, overwrite=True)
        self.st.plot(line_label='template_number', logscale=True,
                     wavelength_min=200, wavelength_max=700, show=False)
        self.st.plot(column_label='template_number',
                     counts=True, aspect_ratio=2, show=False)
        self.st.plot(row_label='template_number', spectral_lines=True,
                     fig_title=r'test $\alpha$', show=False)
        self.st.load_stack(stack_dirname)
        smoothing_bands = self.st.return_smoothing_bands()
        stack = self.st.return_stack()

        self.assertEqual(np.isnan(stack.data).sum(), 0)
        # test shape
        self.assertGreaterEqual(np.prod(stack.data[:, 1, :]), 0)
        self.assertEqual(stack.data.shape[0], len(self.template_numbers))
        norms = np.trapezoid(stack.data[:, 0, :], stack.rf_wl.data, axis=-1)
        self.assertGreater(np.prod(norms), 0)

        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'smoothing_bands.nc')), '300e2041372c479e098f60196e3debc8')

        os.remove(mock_filename)

    @skip_if_frequency_deactivated()
    def test_frequency_evolving_luminosity(self):
        self.sd = 'frequency'
        self.cl = False
        create_mockfile(spectral_density=self.sd, constant_luminosity=self.cl)
        mock_filename = 'mock_catalog_test_frequency_density_evolving_luminosity.csv'
        self.assertTrue(os.path.exists(mock_filename))
        stack_dirname = 'stack_test_frequency_density_normalized'
        flux_units = 'erg / (s cm2 Hz)'

        sys.stdout = None
        self.st.load_catalog(mock_filename, bands_data=repo_home +
                             'filters/test_bands.json', z_label='z', flux_units=flux_units)
        sys.stdout = sys.__stdout__
        self.st.to_rest_frame(flux_conversion='normalized',
                              use_band_responses=True)
        self.st.stack(bin_dict={'template_number==': self.template_numbers},
                      weight='snr_square', error_type='flux_error')
        self.st.stack(
            bin_dict={'template_number==': self.template_numbers}, error_type='std')

        self.st.save_stack(stack_dirname, overwrite=True)
        self.st.plot(line_label='template_number', logscale=True,
                     wavelength_min=200, wavelength_max=700, show=False)
        self.st.plot(column_label='template_number',
                     counts=True, aspect_ratio=2, show=False)
        self.st.plot(row_label='template_number', spectral_lines=True,
                     fig_title=r'test $\alpha$', show=False)
        self.st.load_stack(stack_dirname)
        smoothing_bands = self.st.return_smoothing_bands()
        stack = self.st.return_stack()

        self.assertEqual(np.isnan(stack.data).sum(), 0)
        # test shape
        self.assertGreaterEqual(np.prod(stack.data[:, 1, :]), 0)
        self.assertEqual(stack.data.shape[0], len(self.template_numbers))
        norms = np.trapezoid(stack.data[:, 0, :], stack.rf_wl.data, axis=-1)
        wl_span = stack.rf_wl.data[-1] - stack.rf_wl.data[0]
        self.assertEqual(np.prod(np.isclose(norms, wl_span)), 1)

        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'smoothing_bands.nc')), 'b46b02aee977138730aa0d8c060b5e6c')

        os.remove(mock_filename)


if __name__ == "__main__":
    unittest.main()
