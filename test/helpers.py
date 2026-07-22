import unittest
import os
import sys
import numpy as np
import pandas as pd
import astropy.constants as const
import astropy.units as u

from astropy.cosmology.realizations import Planck18 as cosmo
from stonp.utils import json_loader

cwd = os.getcwd() + '/'
repo_home = cwd

_get_wavelength_def = os.getenv("SKIP_WAVELENGTH_TESTS") == "1"
_get_frequency_def = os.getenv("SKIP_FREQUENCY_TESTS") == "1"
_get_all_def = os.getenv("SKIP_SLOW_TESTS") == "1"


def skip_if_slow_deactivated():
    return unittest.skipIf(_get_all_def,
                           "Skipping slow tests. Remove SKIP_SLOW_TESTS=1 to enable.")


def skip_if_frequency_deactivated():
    return unittest.skipIf(_get_all_def or _get_frequency_def,
                           "Skipping frequency tests. Remove SKIP_FREQUENCY_TESTS=1 to enable.")


def skip_if_wavelength_deactivated():
    return unittest.skipIf(_get_all_def or _get_wavelength_def,
                           "Skipping wavelength tests. Remove SKIP_WAVELENGTH_TESTS=1 to enable.")


def create_mockfile(spectral_density, constant_luminosity):
    """
    Arguments:
    spectral_density : string
    Indicates the kind of spectral sensity file should be created. 
    The acceptable values are 'frequency' and 'wavelength'
    constant_luminosity : bool
    Indicates if luminosity should remain constant along the generated file.
    """
    if not isinstance(spectral_density, str):
        raise TypeError("spectral_density must be a string")
    if not spectral_density in ['frequency', 'wavelength']:
        raise ValueError(
            'spectral_density must be either "wavelength" or "frequency"')
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

    band_names, wl_nb, r_nb, *misc = json_loader(
        repo_home+'filters/test_bands.json')
    template_numbers, _, r_sed, *misc = json_loader(
        repo_home+'spectra/blanton2003_sed_templates.json', sort=False)
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
    else:
        norms_full = np.trapezoid(seds_full, wl_grid_full, axis=1)

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
            select = flux_total >= flux_cut
            df_tmp = df_tmp[select]

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
