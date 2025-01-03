import unittest
import sys
import os
import numpy as np
import scipy.interpolate
import pandas as pd
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo

sys.path.append('../src/stonp/')
import stonp
cwd = os.getcwd() + '/'


class TestJsonLoader(unittest.TestCase):
    def test_no_args(self):
        with self.assertRaises(TypeError):
            stonp.Stacker._json_loader()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            stonp.Stacker._json_loader('test.json')

    def test_bad_file_argument(self):
        with self.assertRaises(TypeError):
            stonp.Stacker._json_loader(False)
            stonp.Stacker._json_loader(1)
            stonp.Stacker._json_loader(1.23)
            stonp.Stacker._json_loader([])

    def test_bad_df_arg(self):
        with self.assertRaises(TypeError):
            stonp.Stacker._json_loader(
                cwd+'../filters/test_bands.json', df=True)
            stonp.Stacker._json_loader(
                cwd+'../filters/test_bands.json', df='a')
            stonp.Stacker._json_loader(cwd+'../filters/test_bands.json', df=7)
            stonp.Stacker._json_loader(
                cwd+'../filters/test_bands.json', df=1.23)
            stonp.Stacker._json_loader(cwd+'../filters/test_bands.json', df=[])

    def test_bad_sort_arg(self):
        with self.assertRaises(TypeError):
            stonp.Stacker._json_loader(
                cwd+'../filters/test_bands.json', sort='a')
            stonp.Stacker._json_loader(
                cwd+'../filters/test_bands.json', sort=1)
            stonp.Stacker._json_loader(
                cwd+'../filters/test_bands.json', sort=1.2)
            stonp.Stacker._json_loader(
                cwd+'../filters/test_bands.json', sort=[])

    def test_returns_len(self):
        self.assertEqual(len(stonp.Stacker._json_loader(
            cwd+'../filters/test_bands.json')), 4)

    def test_returns_type(self):
        nb_labels, wl_nb, r_nb, wl_grid_obs = stonp.Stacker._json_loader(
            cwd+'../filters/test_bands.json')
        self.assertIsInstance(nb_labels, list)
        self.assertIsInstance(wl_nb, np.ndarray)
        self.assertIsInstance(r_nb, scipy.interpolate.interp1d)
        self.assertIsInstance(wl_grid_obs, np.ndarray)


class TestBinDictParser(unittest.TestCase):
    def test_no_args(self):
        with self.assertRaises(TypeError):
            stonp.Stacker._bin_dict_parser()

    def test_bad_dict_arg(self):
        with self.assertRaises(TypeError):
            stonp.Stacker._bin_dict_parser(False)
            stonp.Stacker._bin_dict_parser(1)
            stonp.Stacker._bin_dict_parser(1.23)
            stonp.Stacker._bin_dict_parser([])

    def test_returns_type(self):
        self.assertIsInstance(stonp.Stacker._bin_dict_parser(
            {'test': [1, 2, 3]}), dict)

    def test_dic_struct(self):
        bins = stonp.Stacker._bin_dict_parser(
            {'test1': [1, 2, 3], 'test2': [4, 5, 6]})
        self.assertEqual(len(bins), 2)
        for key in bins:
            for val in bins[key]:
                self.assertIsInstance(val, list)
                self.assertEqual(len(val), 2)

    def test_dict_values(self):
        bins = stonp.Stacker._bin_dict_parser(
            {'test1': [1, 2, 3], 'test2': [4, 5, 6]})
        self.assertListEqual(bins['test1'][0], [1, 2])
        self.assertListEqual(bins['test1'][1], [2, 3])
        self.assertListEqual(bins['test2'][0], [4, 5])
        self.assertListEqual(bins['test2'][1], [5, 6])


class TestGenerator(unittest.TestCase):
    def test_no_args(self):
        with self.assertRaises(TypeError):
            createMockFile()

    def test_only_first_arg(self):
        with self.assertRaises(TypeError):
            createMockFile(spectral_density='wavelength')

    def test_only_second_arg(self):
        with self.assertRaises(TypeError):
            createMockFile(constant_luminosity=True)

    def test_bad_first_arg(self):
        with self.assertRaises(ValueError):
            createMockFile(spectral_density='bad_value',
                           constant_luminosity=True)

        with self.assertRaises(TypeError):
            createMockFile(spectral_density=1, constant_luminosity=True)
            createMockFile(spectral_density=True, constant_luminosity=True)
            createMockFile(spectral_density=1.23, constant_luminosity=True)
            createMockFile(spectral_density=object(), constant_luminosity=True)
            createMockFile(spectral_density=[], constant_luminosity=True)

    def test_bad_second_arg(self):
        with self.assertRaises(TypeError):
            createMockFile(spectral_density='wavelength',
                           constant_luminosity='a')
            createMockFile(spectral_density='wavelength',
                           constant_luminosity=1)
            createMockFile(spectral_density='wavelength',
                           constant_luminosity=1.23)
            createMockFile(spectral_density='wavelength',
                           constant_luminosity=object())
            createMockFile(spectral_density='wavelength',
                           constant_luminosity=[])


class TestStonp(unittest.TestCase):
    st = None
    template_numbers = None

    @classmethod
    def setUpClass(cls):
        for sd in ['wavelength', 'frequency']:
            for cl in [True, False]:
                createMockFile(spectral_density=sd, constant_luminosity=cl)

    @classmethod
    def tearDownClass(cls):
        dirnames = ['stack_test_wavelength_density_luminosity']
        for i in range(4):
            stack_dirname = "stack_test_"
            if i < 2:
                stack_dirname += "wavelength_density_"
            else:
                stack_dirname += "frequency_density_"
            if i % 2 == 0:
                stack_dirname += "luminosity"
            else:
                stack_dirname += "normalized"

            for fname in sorted(os.listdir(os.path.join(cwd, stack_dirname))):
                os.remove(os.path.join(cwd, stack_dirname, fname))
            os.rmdir(stack_dirname)

    def setUp(self):
        self.st = stonp.Stacker()
        template_numbers, _, _, _ = self.st._json_loader(
            cwd+'../spectra/blanton2003_sed_templates.json', sort=False)
        self.template_numbers = [int(template_number)
                                 for template_number in template_numbers]

    def tearDown(self):
        self.st = None
        self.template_numbers = None

    def test_wavelength_constant_luminosity(self):
        mock_filename = 'mock_catalog_test_wavelength_density_constant_luminosity.csv'
        self.assertTrue(os.path.exists(mock_filename))
        stack_dirname = 'stack_test_wavelength_density_luminosity'
        flux_units = 'erg / (s cm2 nm)'

        sys.stdout = None
        self.st.load_catalog(mock_filename, bands_data=cwd +
                             '../filters/test_bands.json', z_label='z', flux_units=flux_units)
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
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'l-template_number.png')), '54653412f4ea262621c128047efa144e')
        self.st.plot(column_label='template_number',
                     counts=True, aspect_ratio=2, show=False)
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 's-template_number_counts.png')), '537a3b25888419f1d757235227c63c44')
        self.st.plot(row_label='template_number', spectral_lines=True,
                     fig_title=r'test $\alpha$', show=False)
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 's-template_number.png')), 'bf3b55fecd23166e96d35e6bb0fec991')
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
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'stacked_seds.nc')), '2847449de4722f8ef473c2c113af44e5')

        os.remove(mock_filename)

    def test_wavelength_evolving_luminosity(self):
        mock_filename = 'mock_catalog_test_wavelength_density_evolving_luminosity.csv'
        self.assertTrue(os.path.exists(mock_filename))
        stack_dirname = 'stack_test_wavelength_density_normalized'
        flux_units = 'erg / (s cm2 nm)'

        sys.stdout = None
        self.st.load_catalog(mock_filename, bands_data=cwd +
                             '../filters/test_bands.json', z_label='z', flux_units=flux_units)
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
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'l-template_number.png')), '347f872e9330a86ba0f92f43beef421f')
        self.st.plot(column_label='template_number',
                     counts=True, aspect_ratio=2, show=False)
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 's-template_number_counts.png')), '4e92011a5e1325d1f6e83d92fc71cb2b')
        self.st.plot(row_label='template_number', spectral_lines=True,
                     fig_title=r'test $\alpha$', show=False)
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 's-template_number.png')), 'a4c4bc1bd45c8977090d52481e8710f4')
        self.st.load_stack(stack_dirname)
        smoothing_bands = self.st.return_smoothing_bands()
        stack = self.st.return_stack()

        self.assertEqual(np.isnan(stack.data).sum(), 0)
        # test shape
        self.assertGreaterEqual(np.prod(stack.data[:, 1, :]), 0)
        self.assertEqual(stack.data.shape[0], len(self.template_numbers))
        norms = np.trapezoid(stack.data[:, 0, :], stack.rf_wl.data, axis=-1)
        wl_span = stack.rf_wl.data[-1] - stack.rf_wl.data[0]
        self.assertGreaterEqual(np.prod(np.isclose(norms, wl_span)), 0)

        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'smoothing_bands.nc')), '86e49f83e3ee938ab1f8843dcbc0cc77')
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'stacked_seds.nc')), 'c13f1cd14aeef1d24aba8ba0747a4cd0')

        os.remove(mock_filename)

    def test_frequency_constant_luminosity(self):
        mock_filename = 'mock_catalog_test_frequency_density_constant_luminosity.csv'
        self.assertTrue(os.path.exists(mock_filename))
        stack_dirname = 'stack_test_frequency_density_luminosity'
        flux_units = 'erg / (s cm2 Hz)'

        sys.stdout = None
        self.st.load_catalog(mock_filename, bands_data=cwd +
                             '../filters/test_bands.json', z_label='z', flux_units=flux_units)
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
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'l-template_number.png')), 'd2e9e3de70b01c206e921d28b03c112d')
        self.st.plot(column_label='template_number',
                     counts=True, aspect_ratio=2, show=False)
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 's-template_number_counts.png')), '537a3b25888419f1d757235227c63c44')
        self.st.plot(row_label='template_number', spectral_lines=True,
                     fig_title=r'test $\alpha$', show=False)
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 's-template_number.png')), '2d348ff0ff923504c695d62269f7c6ba')
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
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'stacked_seds.nc')), 'f23ff64e0a87079b5cf2f34899ca334e')

        os.remove(mock_filename)

    def test_frequency_evolving_luminosity(self):
        mock_filename = 'mock_catalog_test_frequency_density_evolving_luminosity.csv'
        self.assertTrue(os.path.exists(mock_filename))
        stack_dirname = 'stack_test_frequency_density_normalized'
        flux_units = 'erg / (s cm2 Hz)'

        sys.stdout = None
        self.st.load_catalog(mock_filename, bands_data=cwd +
                             '../filters/test_bands.json', z_label='z', flux_units=flux_units)
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
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'l-template_number.png')), '49d10212b1b58b177a6e3accbed14757')
        self.st.plot(column_label='template_number',
                     counts=True, aspect_ratio=2, show=False)
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 's-template_number_counts.png')), '4e92011a5e1325d1f6e83d92fc71cb2b')
        self.st.plot(row_label='template_number', spectral_lines=True,
                     fig_title=r'test $\alpha$', show=False)
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 's-template_number.png')), 'cfb2bb6fb28e296ed865784e8653b38f')
        self.st.load_stack(stack_dirname)
        smoothing_bands = self.st.return_smoothing_bands()
        stack = self.st.return_stack()

        self.assertEqual(np.isnan(stack.data).sum(), 0)
        # test shape
        self.assertGreaterEqual(np.prod(stack.data[:, 1, :]), 0)
        self.assertEqual(stack.data.shape[0], len(self.template_numbers))
        norms = np.trapezoid(stack.data[:, 0, :], stack.rf_wl.data, axis=-1)
        wl_span = stack.rf_wl.data[-1] - stack.rf_wl.data[0]
        self.assertGreaterEqual(np.prod(np.isclose(norms, wl_span)), 0)

        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'smoothing_bands.nc')), '86e49f83e3ee938ab1f8843dcbc0cc77')
        self.assertEqual(calculate_md5(os.path.join(
            cwd, stack_dirname, 'stacked_seds.nc')), '0e28e2e2ad4dddfb570352aeeec5ffe3')

        os.remove(mock_filename)


def createMockFile(spectral_density, constant_luminosity):
    if not isinstance(spectral_density, str):
        raise TypeError("spectral_density must be a string")
    if not isinstance(constant_luminosity, bool):
        raise TypeError("constant_luminosity must be a boolean")
    z_min = 0.1
    z_max = 2
    z_step = 0.01
    n_objs = 10000
    snr_min = 1
    snr_max = 10
    lum_avg = 1e41
    lum_std = 5e40

    band_names, wl_nb, r_nb, _ = stonp.Stacker._json_loader(
        cwd+'../filters/test_bands.json')
    template_numbers, _, r_sed, _ = stonp.Stacker._json_loader(
        cwd+'../spectra/blanton2003_sed_templates.json', sort=False)
    template_numbers = [int(template_number)
                        for template_number in template_numbers]
    rng = np.random.default_rng(seed=996)

    # Drawing redshifts and norms
    z_grid = np.arange(z_min / z_step, z_max / z_step + 1, 1) * z_step
    z_inds = rng.integers(0, len(z_grid), n_objs)
    zs = z_grid[z_inds]

    if constant_luminosity is True:
        lum_avg_real = lum_avg
    else:
        lum_avg_real = lum_avg * (0.5 + 2*zs)

    mu = np.log(lum_avg_real**2 / np.sqrt(lum_std**2 + lum_avg_real**2))
    sigma = np.sqrt(np.log(lum_std**2 / lum_avg**2 + 1))
    lums = rng.lognormal(mu, sigma, size=n_objs)

    # Wavelength grid and normalization in wavelength range
    wl_min = (wl_nb[0] - 10) / (1 + z_max)
    wl_max = (wl_nb[-1] + 10) / (1 + z_min)
    wl_grid_full = np.arange(10*wl_min, 10*wl_max + 1, 1) / 10
    seds_full = r_sed(wl_grid_full)
    if spectral_density == 'frequency':
        fq_grid_full = const.c.to('nm / s').value / wl_grid_full
        seds_full = seds_full * wl_grid_full**2 / const.c.to('nm / s').value
        norms_full = -np.trapezoid(seds_full, fq_grid_full, axis=1)

    elif spectral_density == 'wavelength':
        norms_full = np.trapezoid(seds_full, wl_grid_full, axis=1)
    else:
        raise ValueError(
            'spectral_density must be either "wavelength" or "frequency"')

    # Precomputing nb fluxes for all redshifts
    nb_fluxes_base = np.full(
        [seds_full.shape[0], z_grid.shape[0], len(band_names)], np.nan)
    wl_grid_obs = np.linspace((wl_nb[0] - 10), (wl_nb[-1] + 10), 1000)
    if spectral_density == 'frequency':
        fq_grid_obs = const.c.to('nm / s').value / wl_grid_obs

    for i, z in enumerate(z_grid):
        wl_grid_rest = wl_grid_obs / (1 + z)
        seds = r_sed(wl_grid_rest)
        if spectral_density == 'frequency':
            fq_grid_rest = const.c.to('nm / s').value / wl_grid_rest
            seds = seds * wl_grid_rest**2 / const.c.to('nm / s').value

        responses = r_nb(wl_grid_obs)
        if spectral_density == 'wavelength':
            nb_fluxes_base[:, i, :] = np.trapezoid(
                seds[:, None, :] * responses, wl_grid_obs, axis=-1) / (1 + z)

        elif spectral_density == 'frequency':
            responses /= -np.trapezoid(responses,
                                       fq_grid_obs, axis=-1)[:, None]
            nb_fluxes_base[:, i, :] = -np.trapezoid(
                seds[:, None, :] * responses, fq_grid_obs, axis=-1) * (1 + z)

    # Normalizing so we just need to multiply times luminosity
    nb_fluxes_base /= norms_full[:, None, None]

    # Generating the catalog for each template
    band_error_names = [f'{band_name}_error' for band_name in band_names]
    columns = ['z']
    columns += band_names.copy()
    columns += band_error_names
    columns += ['template_number']

    dfs = []
    dls = cosmo.luminosity_distance(zs).to(u.cm).value

    for n in template_numbers:
        df_tmp = pd.DataFrame(columns=columns)
        # normalizing by luminosity
        nb_fluxes = nb_fluxes_base[n, z_inds, :]
        nb_fluxes *= lums[:, None]
        nb_fluxes /= (4 * np.pi * dls**2)[:, None]
        # Computing errors
        flux_min = nb_fluxes.min()
        flux_max = nb_fluxes.max()
        error_min = flux_min / snr_min
        error_max = flux_max / snr_max
        nb_fluxes_err = (error_min + (nb_fluxes - flux_min) /
                         (flux_max - flux_min) * (error_max - error_min))
        nb_fluxes += rng.normal(0, nb_fluxes_err)
        df_tmp[band_names] = nb_fluxes
        df_tmp[band_error_names] = nb_fluxes_err
        df_tmp.z = zs
        df_tmp.template_number = n

        # If constant_luminosity = False, we will impose a total flux cut
        # based on percentile, to simulate a magnitude cut
        flux_total = np.sum(nb_fluxes, axis=1)
        if constant_luminosity is False:
            flux_cut = np.percentile(flux_total, 5)

        dfs.append(df_tmp)

    df = pd.concat(dfs)
    if constant_luminosity:
        df.to_csv(
            f'mock_catalog_test_{spectral_density}_density_constant_luminosity.csv')
    else:
        df.to_csv(
            f'mock_catalog_test_{spectral_density}_density_evolving_luminosity.csv')


def calculate_md5(file_path):
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    unittest.main()
