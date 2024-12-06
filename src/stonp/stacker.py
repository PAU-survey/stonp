#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Copyright (C) 2022 Pablo Renard
# stonp stacks narrow-band photometric catalogs (https://github.com/PAU-survey/stonp)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.cosmology import Planck18 as cosmo
from astropy import constants as const
from scipy import interpolate


class Stacker():
    '''
    Class to generate stacked SEDs, save them as an `stacked_seds` xarray, 
    and plot them.
    '''

    def __init__(self):
        self.stack_saved = False
        self.alias_dict = {}

    @staticmethod
    def _linterp(x, xp, yp, yp_err):
        """
        Computes linear interpolation and its error, correctly propagated.

        Computes linear interpolation of a 1D function and its error, 
        correctly propagated. Does not accept periodic interpolation. 
        Any interpolated values outside of the original data will be set to NaN

        Parameters
        ----------
        x : array_like
            The x-coordinates at which to evaluate the interpolated values.
        xp : array_like
            The x-coordinates of the data points, must be increasing.
        yp : array_like
            The y-coordinates of the data points, same length as `xp`.
        yp_err : array_like
            The error of the y-coordinates of the data points, same length as
            `xp`.

        Returns
        -------
        y : array_like
            The interpolated values, same shape as `x`.
        y_err : array_like
            The error of the interpolated values, same shape as `x`.

        """

        y = np.interp(x, xp, yp, left=np.nan, right=np.nan)
        # Determining for each point to interpolate the indices of the data points
        # before (1) and after (2)
        ind2 = np.sum(xp < x[:, None], axis=1)
        ind1 = ind2 - 1
        # Setting to 0 the indices >=xp.shape[0]. This is done just to avoid an IndexError.
        # The interpolated error for these points will be set to NaN (as they're outside)
        ind2[ind2 >= xp.shape[0]] = 0
        # Computing derivatives for error propagation
        x1 = xp[ind1]
        x2 = xp[ind2]
        dydy2 = (x - x1) / (x2 - x1)
        dydy1 = 1 - dydy2
        # Propagating error
        y_err1 = yp_err[ind1]
        y_err2 = yp_err[ind2]
        y_err = np.sqrt(dydy1**2 * y_err1**2 + dydy2**2 * y_err2**2)

        y_err[np.isnan(y)] = np.nan

        return y, y_err

    @staticmethod
    def _json_loader(bands_data_dir, df=None, sort=True):
        # Loads the .json of band response functions specified
        # Returns band labels, average wavelengths, and interpolated response
        # functions
        with open(bands_data_dir, 'r') as read_file:
            band_responses_raw = json.load(read_file)

        band_responses = {}
        for key, value in band_responses_raw.items():
            # Removing bands that are not in the catalog, if provided
            if df is not None:
                if key in df.columns:
                    band_responses[key] = value

            # Otherwise we'll add all bands
            else:
                band_responses[key] = value

        for key, value in band_responses.items():
            band_responses[key]['wavelength'] = np.array(value['wavelength'])
            band_responses[key]['response'] = np.array(value['response'])

        # Computing mean wavelengths and sorting in ascending wavelength order
        band_mean_wls = {}
        for key, value in band_responses.items():
            wl = np.array(value['wavelength'])
            r = np.array(value['response'])
            band_mean_wls[key] = np.trapz(wl * r, wl) / np.trapz(r, wl)

        if sort:
            inds = np.argsort(list(band_mean_wls.values()))

            band_mean_wls = {list(band_mean_wls.keys())[i]: list(band_mean_wls.values())[i]
                             for i in inds}

            band_responses = {list(band_responses.keys())[i]: list(band_responses.values())[i]
                              for i in inds}

        # Generating interpolation object
        # returns all interpolated normalized bands for a given wavelength grid, at once
        # Computing the highest resolution common wavelength grid for all band responses
        wl_bands = [value['wavelength'] for value in band_responses.values()]
        wl_bands_min = np.min([wl_band[0] for wl_band in wl_bands])
        wl_bands_max = np.max([wl_band[-1] for wl_band in wl_bands])
        wl_bands_step = np.min(
            [np.min(wl_band[1:] - wl_band[:-1]) for wl_band in wl_bands])
        wl_grid_bands = np.arange(wl_bands_min / wl_bands_step,
                                  wl_bands_max / wl_bands_step + 1) * wl_bands_step

        # Interpolating to a single array the band responses
        r_nb = np.zeros([len(band_responses), wl_grid_bands.shape[0]])
        i = 0
        for value in band_responses.values():
            r_nb[i, :] = np.interp(wl_grid_bands, value['wavelength'], value['response'],
                                   left=0, right=0)
            r_nb[i, :] /= np.trapz(r_nb[i, :], wl_grid_bands)
            i += 1

        # Computing the interpolation object
        r_nb = interpolate.interp1d(
            wl_grid_bands, r_nb, bounds_error=False, fill_value=0)

        nb_labels = list(band_mean_wls.keys())
        wl_nb = np.array(list(band_mean_wls.values()))

        return nb_labels, wl_nb, r_nb

    @staticmethod
    def _bin_dict_parser(bin_dict):
        # Parses the bin_dict so all bins are nested lists of two elements

        for key, bin_edgs in bin_dict.items():
            if key[-2:] != '==':
                if not isinstance(bin_edgs[0], (tuple, list)):
                    bin_edgs = [bin_edgs]

                bin_edgs_new = list()
                for bin_edg in (bin_edgs):
                    for i in range(len(bin_edg) - 1):
                        bin_edgs_new.append([bin_edg[i], bin_edg[i+1]])

                bin_edgs = bin_edgs_new

                bin_dict[key] = bin_edgs

        return bin_dict

    @staticmethod
    def _determine_cols_rows(n_subplots, aspect_ratio):
        # Determines the number of columns a rows for a plot with a given
        # number of subplots and aspect ratio (assmuming all subplots square)

        n_cols = int(np.sqrt(n_subplots) * aspect_ratio)
        n_rows = int(np.sqrt(n_subplots) / aspect_ratio)
        while n_rows * n_cols < n_subplots:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        return n_cols, n_rows

    @staticmethod
    def _single_plotter(ax, stacked_seds_tmp, kw, line_label=None,
                        xlabel=None, ylabel=None, legend_labels=None, title=None,
                        extra_xlabel=None, extra_ylabel=None, counts=False,
                        spectral_lines_dict=None, spectral_lines_legend=False,
                        logscale=False, sharey=False):
        # Makes a stacked SED plot on a given axis. Check plot() to understand
        # entry parameters

        x = stacked_seds_tmp['rf_wl'].data
        if line_label:
            n_lines = stacked_seds_tmp[line_label].shape[0]
        else:
            n_lines = 1

        for k in range(n_lines):
            color = f'C{k}'
            if line_label:
                kw[line_label] = k

            if counts:
                y = stacked_seds_tmp.isel(**kw).sel(data='counts')
                ax.step(x, y, color=color, where='mid')
            else:
                y = stacked_seds_tmp.isel(**kw).sel(data='flux')
                y_err = stacked_seds_tmp.isel(**kw).sel(data='flux_error')
                ax.plot(x, y, color=color)
                ax.fill_between(x, y+y_err, y-y_err, color=color,
                                alpha=0.3, label='_nolegend_')

        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.tick_params(which='both', bottom=False)

        if ylabel:
            ax.set_ylabel(ylabel)
        elif sharey:
            ax.tick_params(which='both', left=False)

        if legend_labels:
            if spectral_lines_legend:
                leg1 = ax.legend(legend_labels, loc='upper right')
            else:
                leg1 = ax.legend(legend_labels)

        if title:
            ax.set_title(title)

        if extra_xlabel:
            ax_twiny = ax.twiny()
            ax_twiny.set_xlabel(extra_xlabel)
            ax_twiny.tick_params(which='both', top=False, bottom=False)
            ax_twiny.set_xticks([])
            ax.tick_params(which='both', top=False, bottom=False)

        if extra_ylabel:
            ax_twinx = ax.twinx()
            ax_twinx.set_ylabel(extra_ylabel)
            ax_twinx.tick_params(which='both', right=False, left=False)
            ax_twinx.set_yticks([])
            if sharey:
                ax.tick_params(which='both', right=False, left=False)

        if logscale:
            ax.set_yscale('log')

        ax.set_xlim(x[0], x[-1])

        if spectral_lines_dict:
            for key, wls in spectral_lines_dict.items():
                if not isinstance(wls, (tuple, list)):
                    spectral_lines_dict[key] = [wls]

            n = 0
            lines = []
            for key, wls in spectral_lines_dict.items():
                color = f'C{9-n}'
                n += 1
                same_line = False
                for wl in wls:
                    line = ax.axvline(
                        wl, color=color, linewidth=1, linestyle='-.')
                    if not same_line:
                        lines.append(line)
                        same_line = True

            if spectral_lines_legend:
                if legend_labels:
                    ax.legend(lines, list(spectral_lines_dict.keys()),
                              loc='lower center')

                else:
                    ax.legend(lines, list(spectral_lines_dict.keys()))

                ax.add_artist(leg1)

    @staticmethod
    def _rc_parameters(rc_params=None):
        # Defines the default parameters for plotting layout with matplotlib

        plt.rcParams.update({'axes.labelsize': 'large', 'axes.titlesize': 'large',
                             'xtick.labelsize': 'large', 'ytick.labelsize': 'large',
                             'xtick.minor.visible': True, 'ytick.minor.visible': True,
                             'xtick.major.size': 5, 'xtick.major.width': 1, 'xtick.minor.size': 3,
                             'ytick.major.size': 5, 'ytick.major.width': 1, 'ytick.minor.size': 3,
                             'figure.figsize': [6.4, 6.4/1.62], 'figure.dpi': 200,
                             'legend.fontsize': 'large',
                             'text.usetex': False})

        if rc_params:
            plt.rcParams.update(rc_params)

    @staticmethod
    def _query_yes_no(question, default="yes"):
        """Ask a yes/no question via raw_input() and return their answer.

        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
                It must be "yes" (the default), "no" or None (meaning
                an answer is required of the user).

        The "answer" return value is True for "yes" or False for "no".
        """
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == "":
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write(
                    "Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


    def _get_alias(self, label):
        # returns the plotting alias of a label, if specified before

        try:
            alias_ = self.alias_dict[label]
        except:
            alias_ = label

        return alias_

    def _range_label(self, label, index):
        # Returns a latex string with the value range of a given bin

        alias = self._get_alias(label)
        if f'{label}_min' in self.stacked_seds.attrs:
            bin_edg_min = self.stacked_seds.attrs[f'{label}_min'][index]
            bin_edg_max = self.stacked_seds.attrs[f'{label}_max'][index]
            range_label = rf'{bin_edg_min:.4g}$\leq${alias}<{bin_edg_max:.4g}'

        else:
            bin_mid = self.stacked_seds.coords[label].data[index]
            range_label = rf'{alias}={bin_mid}'

        return range_label

    def define_aliases(self, alias_dict):
        '''
        Defines the plotting aliases for catalog labels.

        Defines the plotting aliases for catalog labels (e.g., `zb`, `sfr_log`,
        etc.) Will only be used for plotting, not filenames. Accepts LaTeX
        format.

        Parameters
        ----------
        alias_dict : dict
            Dictionary of aliases. Keys are catalog column labels, values are 
            the aliases. Not all the column labels need to be in the dictionary.

        Returns
        -------
        None.

        '''

        self.alias_dict = alias_dict

    def load_catalog(self, catalog, max_nan_bands=0, z_label='zb',
                     fill_nans='interpolated', bands_data=None,
                     bands_error_suffix='_error', flux_units='erg / (s cm2 nm)',
                     wavelength_units='nm'):
        '''
        Loads a photometric galaxy catalog to stack.

        Loads a photometric galaxy catalog to stack. The catalog must have
        exactly one row per object.

        Parameters
        ----------
        catalog : pandas DataFrame or str
            If DataFrame instance, the catalog itself, already loaded.
            If str, directory of the catalog to be loaded.
            Catalogs can be in plain text (.csv) or .parquet
        max_nan_bands : int, optional
            Maximum number of NaN band fluxes that can be tolerated. Objects
            with more NaN bands will be removed from the catalog when loading.
            If inhomogeneous bands (broad and narrow) are present in the 
            catalog, it is advisable to set it to 0. The default is 0.
        z_label : str, optional
            The label of the redshift column to be used. The default is 'zb'.
        fill_nans: 'interpolated', 'zero'
            Interpolate NaN flux values or set them to zero. Setting them to
            zero is not advised if flux_conversion='normalized' when running
            to_rest_frame(). Errors are always interpolated.
            The default is 'interpolated'
        bands_data: dict or str, optional
            If dict, dictionary with photometric band wavelength data. 
            Keys must be the column labels, values their average wavelengths. 
            If str, directory of the .json file that contains all band response
            functions. The .json must contain a dictionary where the keys are
            the column labels of the catalog, and the values are the response
            functions. These must be specified as dictionaries with 
            two elements: 'response' (list of the throughput) and 'wavelength'
            (list of the respective wavelengths). 'wavelength' does not need to
            be the same for all bands in the .json.
            Will use default PAUS average wavelengths if None. 
            The default is None.
        bands_error_suffix : str, optional
            Suffix to be appended to the band flux column labels in order to 
            refer to the band flux error columns. The default is '_error'.
        flux_units : str or astropy `unit` object, optional
            The units of the band fluxes. Can be either spectral flux 
            wavelength density or spectral flux frequency density. If string, 
            must be compatible with the `astropy.units` string format. 
            The default is 'erg / (s cm2 nm)'.
        wavelength_units : str or astropy `unit` object, optional
            The wavelength units of the bands. If string, must be
            compatible with the `astropy.units` string format. Does not need to
            match the wavelength units of `flux_units`. The default is 'nm'.

        Returns
        -------
        None.

        '''

        self.max_nan_bands = max_nan_bands
        self.z_label = z_label
        self.flux_units_catalog = u.Unit(flux_units)
        self.wavelength_units = u.Unit(wavelength_units)

        # Checking if flux units are wavelength density or frequency density
        if 'spectral flux density wav' in self.flux_units_catalog.physical_type:
            self.flux_density = 'wavelength'
            unit_bases = self.flux_units_catalog.bases
            unit_powers = self.flux_units_catalog.powers
            for unit_base, unit_power in zip(unit_bases, unit_powers):
                if unit_base.physical_type == 'length' and unit_power ==-1:
                    self.wavelength_flux_units = unit_base

        elif 'spectral flux density' in self.flux_units_catalog.physical_type:
            self.flux_density = 'frequency'
            unit_bases = self.flux_units_catalog.bases
            for unit_base in unit_bases:
                if unit_base.physical_type == 'frequency':
                    self.frequency_units = unit_base


        else:
            raise ValueError(
                "Flux units not recognized as spectral flux density in wavelength or frequency")

        if isinstance(catalog, pd.DataFrame):
            df = catalog.copy(deep=True)

        else:
            try:
                df = pd.read_csv(catalog)

            except UnicodeDecodeError:
                df = pd.read_parquet(catalog)

        print(f'Objects in catalog: {len(df)}')

        if not bands_data:
            self.nb_labels = ['NB455', 'NB465', 'NB475', 'NB485', 'NB495', 'NB505',
                              'NB515', 'NB525', 'NB535', 'NB545', 'NB555', 'NB565',
                              'NB575', 'NB585', 'NB595', 'NB605', 'NB615', 'NB625',
                              'NB635', 'NB645', 'NB655', 'NB665', 'NB675', 'NB685',
                              'NB695', 'NB705', 'NB715', 'NB725', 'NB735', 'NB745',
                              'NB755', 'NB765', 'NB775', 'NB785', 'NB795', 'NB805',
                              'NB815', 'NB825', 'NB835', 'NB845']

            self.wl_nb = np.array([457.80120531, 467.9232592, 476.12485344, 485.53838507,
                                   496.22121905, 506.31404877, 516.6699263, 525.72367734,
                                   535.98739846, 546.81899053, 556.77200626, 566.03474008,
                                   575.64197073, 586.32069947, 596.12556457, 605.74386357,
                                   615.42604602, 626.09265349, 636.34222135, 645.21309411,
                                   656.03432415, 665.73663592, 676.31223727, 685.6460025,
                                   695.44801782, 705.98868673, 715.64945529, 725.93512278,
                                   735.52350426, 745.49322613, 755.29712726, 766.76179833,
                                   775.08781748, 784.74748909, 795.16537091, 805.04549444,
                                   815.15565658, 825.48236959, 835.88372417, 845.84031742])

        elif isinstance(bands_data, dict):
            self.nb_labels = list(bands_data.keys())
            self.wl_nb = np.array(list(bands_data.values()))

        elif isinstance(bands_data, str):
            nb_labels, wl_nb, r_nb = self._json_loader(bands_data, df=df)

            self.nb_labels = nb_labels
            self.wl_nb = wl_nb
            self.r_nb = r_nb

        else:
            raise ValueError(
                "'bands_data' must be a string (directory of .json file) or dict")

        self.nb_err_labels = [
            label + bands_error_suffix for label in self.nb_labels]

        # Removing objects with more nan bands than max_nan_bands
        select = np.sum(
            np.isnan(df[self.nb_labels].values), axis=1) <= max_nan_bands
        print(f'Objects removed because of too many NaN bands: {np.sum(~select)}',
              f'({np.sum(~select) / len(df) * 100:.2f} %)')
        df = df[select]

        # Removing objects without redshift (NaN)
        df = df[~np.isnan(df[self.z_label])]

        print(f'Objects after removal: {len(df)}')

        # Interpolating remaining NaN bands, unless they're the bluest/reddest
        seds = df[self.nb_labels].values
        seds_err = df[self.nb_err_labels].values
        select = np.sum(np.isnan(seds), axis=1) > 0
        inds = select.nonzero()[0]
        for ind in inds:
            nans = np.isnan(seds[ind, :])
            if np.sum(nans) > 0:
                seds[ind, :], seds_err[ind, :] = self._linterp(self.wl_nb, self.wl_nb[~nans],
                                                               seds[ind, ~nans], seds_err[ind, ~nans])

                if fill_nans.lower() == 'zeros':
                    seds[ind, nans] = 0

        df[self.nb_labels] = seds
        df[self.nb_err_labels] = seds_err
        self.df = df

    def load_stack(self, stack_folder):
        '''
        Loads a `stacked_seds` xarray, previously saved as a netCDF file (.nc).

        Parameters
        ----------
        stack_folder : str
            Directory of the `stacked_seds` xarray to be loaded.

        Returns
        -------
        None.

        '''
        self.stack_folder = f'{stack_folder}/'
        self.stacked_seds = xr.open_dataarray(
            f'{self.stack_folder}stacked_seds.nc')
        self.stack_saved = True

    def save_stack(self, stack_folder, overwrite=False):
        '''
        Saves the current `stacked_seds` as an xarray (netCDF file, .nc).

        Parameters
        ----------
        stack_folder : str
            Directory of the stacked_seds xarray to be saved.
        overwrite: bool, optional
            Overwrites without asking the stacked_seds.nc in the `stacked_seds`
            folder. Default is False.

        Returns
        -------
        None.

        '''
        os.makedirs(stack_folder, exist_ok=True)
        self.stack_folder = f'{stack_folder}/'
        if os.path.exists(f'{self.stack_folder}stacked_seds.nc') and not overwrite:
            answer = self._query_yes_no(f'stacked_seds.nc in {stack_folder} '
                                        'already exists. Overwrite?')

            if answer == True:
                self.stacked_seds.to_netcdf(
                    f'{self.stack_folder}stacked_seds.nc')
                self.stack_saved = True
            else:
                print('Current stack was not saved. Please change stack_folder or'
                      f' manually delete f{stack_folder}/stacked_seds.nc')

        else:
            self.stacked_seds.to_netcdf(f'{self.stack_folder}stacked_seds.nc')
            self.stack_saved = True

    def return_stack(self):
        '''
        Returns the current 'stacked_seds'

        Returns
        -------
        stacked_seds : xarray
            The xarray object with the stacked_seds and all the relevant
            metadata.

        '''

        return self.stacked_seds

    def column_histogram(self, label,  bins, save=False):
        # why does this return n obj min and its value?
        # can't we just have the histogram as it is? Doesn't seem used elsewhere
        """
        Plots a histogram of a given column of the loaded catalog.

        Plots a histogram of a given column of the loaded catalog, with a given 
        uniform binning. A photometric catalog needs to have been loaded before.
        Will also return value and number of objects of the bin(s) with the 
        least objects. Will be saved in `stack_folder` if the `stacked_seds` 
        xarray has been saved, or in the working directory otherwise.

        Parameters
        ----------
        label : str
            Label of the column to be used as data for the histogram.
        bins : int or array_like
            Number of bins, or array_like with the bin edges.
        save_plot : bool, optional
            Saves the plot as .png and .pdf in stack_folder (or working
            directory).

        Returns
        -------
        n_obj_min : int
            Number of objects in the bin with the least objects.
        value_obj_min : float
            Mean value of the bin with the least objects.

        """

        try:
            self.df
        except AttributeError:
            raise Exception(
                "No catalog loaded. Please run load_catalog() first.")

        plt.figure(dpi=160, figsize=[6.4, 6.4 / 1.62])
        counts, bin_edg = plt.hist(self.df[label], bins=bins)[:-1]
        bin_mid = bin_edg[1:]
        plt.xlabel(self._get_alias(label))
        plt.ylabel('N obj')
        if save:
            if self.saved_stack:
                folder = self.stack_folder
            else:
                folder = ''

            plt.savefig(f'{folder}{label}_hist.pdf',
                        bbox_inches='tight', transparent=True)
            plt.savefig(f'{folder}{label}_hist.png',
                        bbox_inches='tight', transparent=True)

        plt.show()

        n_obj_min = counts.min()
        value_obj_min = bin_mid[np.nonzero(counts == n_obj_min)[0]]

        return n_obj_min, value_obj_min

    def to_rest_frame(self, flux_conversion='normalized', use_band_responses=False,
                      wl_rf_step=1, wl_obs_min=None, wl_obs_max=None,
                      z_min=None, z_max=None, compute_error=True,
                      n_bins_lum_vs_z=20, show_lum_plot=True,
                      save_lum_plot=False):
        """
        Shifts all band fluxes to rest frame, with a given unit conversion.

        Shifts all band fluxes to rest frame, using a given unit conversion for
        the photometric fluxes.

        Parameters
        ----------
        flux_conversion : str, optional
            The unit conversion applied with the rest-frame shifting. Possible
            values are:

            - 'normalized': Normalized all SEDs so their integral is equal
              to their wavelength span.
            - 'luminosity': Converts from spectral flux density to spectral
              luminosity density.
            - 'nothing': Does not apply any unit conversion, shifts fluxes
              as they are in the catalog.

            The default is 'normalized'.
        use_band_responses: bool, optional
            If True, will shift SEDs to the rest-frame wavelength grid using 
            a sum weighted by the integral of the response functions. 
            If False, will shift to the rest-frame grid applying linear
            interpolation (assuming each band flux is located exactly at the
            mean wavelength of the band). This is faster, but less accurate, and
            does not allow the use of a inhomogeneous set of bands (i.e., bands
            of significantly different FWHM, or bands with an irregular
            spectral coverage.)
            The default is False.
        wl_rf_step : float, optional
            Step between points of the rest-frame uniform wavelength grid, in
            the specified wavelength units. The default is 1.
        wl_obs_min: float, optional
            Minimum observed frame wavelength to be considered. If None, will
            be the smallest mean wavelength of the bands. The default is None.
        wl_obs_max: float, optional
            Maximum observed frame wavelength to be considered. If None, will
            be the largest mean wavelength of the bands. The default is None.
        z_min : float, optional
            Minimum redshift to be considered. If not given, will be the minimum
            redshift of the catalog.
        z_max : float, optional.
            Maximum redshift to be considered. If not given, will be the maximum
            redshift of the catalog.
        compute_error : bool, optional
            Determine if the error must be determined or not when shifting
            to rest frame. Setting it to False will significantly reduce 
            computation time, but will not allow the use of any specific
            weighting when stacking, nor computing error-based shaded areas 
            when running `Stacker.stack()` or `Stacker.plot()`.
            Default is True.

        Returns
        -------
        None.

        """

        try:
            self.df
        except AttributeError:
            raise Exception(
                "No catalog loaded. Please run load_catalog() first.")

        flux_conversion = flux_conversion.lower()
        if flux_conversion not in ['normalized', 'redshift_normalized', 'luminosity', 'nothing']:
            raise Exception(
                "flux_conversion not understood. Please input 'normalized', 'luminosity' or 'nothing'")

        if use_band_responses:
            try:
                self.r_nb
            except:
                raise Exception("Response functions of the photometric bands not loaded. "
                                "Please run 'load_catalog' specifying in 'bands_data' the directory "
                                "of the .json file that contains the response function data ")

        if not z_min:
            z_min = self.df[self.z_label].min()

        if not z_max:
            z_max = self.df[self.z_label].max()

        if not wl_obs_min:
            wl_obs_min = self.wl_nb[0]

        if not wl_obs_max:
            wl_obs_max = self.wl_nb[-1]
            
        # Removing all objects outside of redshift range
        select_z = ((self.df[self.z_label].values >= z_min)
                    & (self.df[self.z_label].values <= z_max))

        # rest-frame wavelength grid
        scaling_wl = 1 / wl_rf_step  # scaling to integers first
        wl_rf_min = wl_obs_min / (1 + z_max)
        wl_rf_max = wl_obs_max / (1 + z_min)
        wl_grid = np.arange(wl_rf_min * scaling_wl,
                            wl_rf_max * scaling_wl + 1, 1)
        wl_grid /= scaling_wl            
        if self.flux_density == 'frequency':
            fq_grid = (wl_grid * self.wavelength_units).to(
                self.frequency_units, equivalencies=u.spectral())
            fq_grid = fq_grid.value
        
        
        df_tmp = self.df[select_z]
        rf_seds = np.zeros([len(df_tmp), wl_grid.shape[0]])
        if compute_error:
            rf_seds_err = np.zeros([len(df_tmp), wl_grid.shape[0]])
        seds = df_tmp[self.nb_labels].values
        seds_err = df_tmp[self.nb_err_labels].values
        zs = df_tmp[self.z_label].values
        #progress_old = 0
        for i in range(rf_seds.shape[0]):

            sed = seds[i, :]
            sed_err = seds_err[i, :]
            z = zs[i]
            select_wl_obs = (wl_grid >= wl_obs_min / (1 + z)
                             ) & (wl_grid <= wl_obs_max / (1 + z))
            if use_band_responses:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="invalid value encountered in divide")
                    nb_weights = self.r_nb(wl_grid * (1 + z))**2
                    # NaN bands won't count at all
                    nb_weights[np.isnan(sed), :] = 0
                    rf_sed = np.nansum(
                        nb_weights * sed[:, None], axis=0) / np.nansum(nb_weights, axis=0)
                    if compute_error:
                        rf_sed_err = np.sqrt(
                            np.nansum((nb_weights * sed_err[:, None])**2, axis=0))
                        rf_sed_err /= np.nansum(nb_weights, axis=0)

                    # Taking into account only the wavelengths inside (wl_obs_min, wl_obs_max)
                    rf_sed[~select_wl_obs] = np.nan
                    if compute_error:
                        rf_sed_err[~select_wl_obs] = np.nan

            else:
                if compute_error:
                    rf_sed, rf_sed_err = self._linterp(wl_grid, self.wl_nb / (1+z),
                                                       sed, sed_err)
                else:
                    rf_sed = np.interp(wl_grid, self.wl_nb / (1+z), sed,
                                       left=np.nan, right=np.nan)

                rf_sed[~select_wl_obs] = np.nan
                if compute_error:
                    rf_sed_err[~select_wl_obs] = np.nan

            # Re-scaling flux densities as they are now in rest frame (to conserve bolometric flux)
            # rest-frame scaling
            if self.flux_density == 'wavelength':
                rf_sed *= (1 + z)
                if compute_error:
                    rf_sed_err *= (1 + z)

            elif self.flux_density == 'frequency':
                rf_sed /= (1 + z)
                if compute_error:
                    rf_sed_err /= (1 + z)

            if flux_conversion == 'normalized':
                self.flux_conversion = 'normalized'
                # negative fluxes set to zero to avoid normalization issues
                rf_sed[rf_sed < 0] = 0
                select = ~np.isnan(rf_sed)
                if self.flux_density == 'wavelength':
                    # Since we're normalizing, we don't care if wavelength units
                    # of flux 
                    norm = np.trapz(rf_sed[select], wl_grid[select])
                    wl_span = wl_grid[select][-1] - wl_grid[select][0]
                    rf_sed = rf_sed / norm * wl_span  # norm of rest-frame SED equal to wavelength span
                    if compute_error:
                        rf_sed_err = rf_sed_err / norm * wl_span

                elif self.flux_density == 'frequency':
                    norm = -np.trapz(rf_sed[select], fq_grid[select])
                    fq_span = fq_grid[select][0] - fq_grid[select][-1]
                    rf_sed = rf_sed / norm * fq_span  # norm of rest-frame SED equal to wavelength span
                    if compute_error:
                        rf_sed_err = rf_sed_err / norm * fq_span

            rf_seds[i, :] = rf_sed
            if compute_error:
                rf_seds_err[i, :] = rf_sed_err

            #progress = i / rf_seds.shape[0] * 100
            # if progress - progress_old > 10:
            #    print(f'{progress:.0f} %')
            #    progress_old = progress

        # Converting rf_seds to masked array
        rf_seds = np.ma.array(rf_seds, mask=np.isnan(rf_seds))
        if compute_error:
            rf_seds_err = np.ma.array(rf_seds_err, mask=np.isnan(rf_seds))

        # if flux_conversion = 'luminosity' or flux_conversion == 'redshfit_normalized',
        # computing luminosity distances anc converting fluxes to luminosities
        if flux_conversion == 'luminosity' or flux_conversion == 'redshift_normalized':
            distance_ind = self.flux_units_catalog.powers.index(-2)
            distance_units = self.flux_units_catalog.bases[distance_ind]

            # from flux to luminosity
            dl = cosmo.luminosity_distance(zs).to(distance_units)
            rf_seds *= 4 * np.pi * dl[:, None].value**2
            if compute_error:
                rf_seds_err *= 4 * np.pi * dl[:, None].value**2

            # If flux_conversion == 'luminosity', we stop here
            if flux_conversion == 'luminosity':
                self.flux_units = self.flux_units_catalog * dl.unit**2

            # If flux_conversion == 'redshift normalized', we'll determine average
            # luminosity vs z and divide all fluxes by average luminosity for their z
            elif flux_conversion == 'redshift_normalized':
                # Computing mean luminosity per wide redshift bin
                if self.flux_density == 'wavelength':
                    lums = np.trapz(rf_seds, wl_grid, axis=1).data
                    
                elif self.flux_density == 'frequency':    
                    lums = np.trapz(rf_seds, fq_grid, axis=1).data

                z_bin_edg = np.linspace(z_min, z_max, n_bins_lum_vs_z+1)
                z_bin_mid = (z_bin_edg[1:] + z_bin_edg[:-1]) / 2

                lums_avg = np.zeros(z_bin_mid.shape[0])
                lums_sq_err = np.zeros(lums.shape[0])
                for i, _ in enumerate(z_bin_mid):
                    select = (zs >= z_bin_edg[i]) * (zs < z_bin_edg[i+1])
                    lums_avg[i] = np.ma.mean(lums[select])
                    lums_sq_err[select] = (lums[select] - lums_avg[i])**2

                lums_std = np.sqrt(np.sum(lums_sq_err) / len(lums))

                # Computing interpolation spline (s according to scipy prescription)
                s_spline = len(lums_avg)*lums_std**2
                spline_params = interpolate.splrep(z_bin_mid, lums_avg,
                                                   s=s_spline, k=5)

                # Cropping objects with z lower than first bin midpoint,
                # or higher than last bin midpoint, to avoid interpolation issues
                # z_min and z_max will be modified accordingly
                z_min = z_bin_mid[0]
                z_max = z_bin_mid[-1]
                
                # Cropping wavelenght grid to new redshift limits
                wl_ind_min = np.argmin(np.abs(wl_grid - wl_obs_min / (1 + z_max)))
                wl_ind_max = np.argmin(np.abs(wl_grid - wl_obs_max / (1 + z_min)))
                wl_grid = wl_grid[wl_ind_min:wl_ind_max+1]
                
                select_z = (zs >= z_min) & (zs <= z_max)
                rf_seds = rf_seds[select_z, wl_ind_min:wl_ind_max+1]
                if compute_error:
                    rf_seds_err = rf_seds_err[select_z, wl_ind_min:wl_ind_max+1]
                    
                zs = zs[select_z]
                dl = dl[select_z]
                lums = lums[select_z]
                
                print(f'z_min set to {z_min:.3g}, z_max set to {z_max:.3g}'
                      ,'\nto avoid extrapolation errors for average luminosity versus redshift')
                
                # Computing luminosity for each redshift
                lums_vs_z = lums_vs_z = interpolate.BSpline(*spline_params)(zs)



                # plotting if specified
                if show_lum_plot or save_lum_plot:
                    lum_min, lum_max = np.percentile(lums, [0.5, 99.9])
                    if lum_min <= 0:
                        lum_min = np.ma.median(lums) / 100

                    z_grid_plot = np.linspace(z_min, z_max, 200)
                    lum_vs_z_plot = interpolate.BSpline(*spline_params)(z_grid_plot)

                    plt.figure(figsize=[8, 8/1.62], dpi=200)
                    plt.plot(z_bin_mid, lums_avg, marker='o', linestyle=':',
                             color='tab:orange', label=r'$\langle D_L \rangle_z$')
                    plt.plot(z_grid_plot, lum_vs_z_plot,
                             label=r'Interpolated $D_L(z)$', color='tab:orange')

                    x = zs
                    y = lums
                    bins = 1000
                    data, x_e, y_e = np.histogram2d(
                        x, y, bins=bins, density=True)
                    z = interpolate.interpn((0.5*(x_e[1:] + x_e[:-1]),
                                             0.5*(y_e[1:]+y_e[:-1])),
                                            data, np.vstack([x, y]).T,
                                            method="splinef2d", bounds_error=False)

                    # Sorting objects so the densest are plotted last
                    idx = z.argsort()
                    x, y, z = x[idx], y[idx], z[idx]
                    plt.scatter(x, y, s=0.5, c=z, cmap='viridis')
                    plt.legend(loc='upper left')
                    plt.xlabel('z')
                    plt.ylim(lum_min, lum_max)
                    plt.xlim(z_min, z_max)
                    if self.flux_density == 'wavelength':
                        lum_units = (self.flux_units_catalog * dl.unit**2 
                                     * self.wavelength_flux_units)
                    
                    elif self.flux_density == 'frequency':
                        lum_units = (self.flux_units_catalog * dl.unit**2 
                                     * self.frequency_units)
                        
                    plt.ylabel(
                        f'Observed luminosity [{lum_units:latex_inline}]')
                    plt.yscale('log')
                    if save_lum_plot:
                        plt.savefig('average_luminosity_vs_redshift.pdf',
                                    transparent=True)
                        plt.savefig('average_luminosity_vs_redshift.png',
                                    transparent=True)

                    if show_lum_plot:
                        plt.show()
                    else:
                        plt.close()

                # Normalizing fluxes by luminosity per redshift
                rf_seds /= lums_vs_z[:, None]
                if compute_error:
                    rf_seds_err /= lums_vs_z[:, None]

                self.flux_units = u.dimensionless_unscaled

        # other flux conversion cases
        elif flux_conversion == 'normalized':
            self.flux_units = u.dimensionless_unscaled

        elif flux_conversion == 'nothing':
            self.flux_units = self.flux_units_catalog

        self.rf_seds = rf_seds
        if compute_error:
            self.rf_seds_err = rf_seds_err

        self.wl_grid = wl_grid
        self.flux_conversion = flux_conversion
        self.wl_obs_min = wl_obs_min
        self.wl_obs_max = wl_obs_max
        self.z_min = z_min
        self.z_max = z_max
        self.zs = zs

    def stack(self, bin_dict={}, weight=None, error_type=None, min_n_obj=0):
        '''
        Stacks the rest-frame SEDs into the specified bins, and stores them.

        Stacks the rest-frame SEDs into the specified bins by computing their
        averages, then stores them as a multi-dimensional and labelled xarray 
        object (`stacked_seds`).

        Parameters
        ----------
        bin_dict : dict, optional
            A dictionary specifying all the bins where the SEDs will be stacked.
            Each key must be a column label of the catalog, and each value a 
            list of bin edges. These can be a list of continuous edges 
            (with n_bins+1 elements), or a list of lists of length 2, where 
            each sub-list contains the lower and upper edges of a given bin.
            Mixing both binning input types is accepted.
            If a key is composed of `column_label` + '%%', the bin edges will be 
            assumed to be percentiles. The trailing '%%' will be removed before
            querying the dataframe.
            If a key is composed of `column_label` + '==', the bins will be assumed
            to be discrete (i.e., each bin will be defined as `bin_values` == `bin_edg`, 
            not `bin_edg` [n]<= `bin_values` < `bin_edg` [n+1]). Only a non-nested list of
            discrete bin values will be accepted in this case. The trailing '==' 
            will be removed before querying the dataframe.
            The default is {} (no binning, all SEDs stacked into one).
        weight : str, optional
            Weighting to be used when stacking the SEDs. Can be 'inv_variance' 
            or 'snr_square'. The default is None.
        error_type : str, optional
            The type of error of the stacked SEDs. Can be 'flux_error' (the 
            propagated flux error of all stacked objects), 'std' (the standard
            deviation of the flux) or 'std_mean' (the standard deviation of
            the mean of the flux). The default is None.
        min_n_obj : int, optional
            Minimum number of objects per wavelength point for a stack to be
            considered valid. Wavelengths with less objects will be masked.
            The default is 0.

        Returns
        -------
        None.

        '''

        # Pasing the input and checking that everything has been computed
        try:
            self.rf_seds
        except AttributeError:
            raise Exception(
                'SEDs not shifted to rest frame. Please run to_rest_frame() first')

        weights = ['inv_variance', 'snr_square']
        if weight and weight not in weights:
            raise ValueError(f"{weight} is not a valid weight specification."
                             " Please input 'inv_variance' or 'snr_square'")

        error_types = ['flux_error', 'std', 'std_mean']
        if error_type and error_type not in error_types:
            raise ValueError(f"{error_type} is not a valid error type specification."
                             " Please input 'flux_error', 'std', 'std_mean' or None")

        use_errors = False
        if weight or str(error_type).lower() == 'flux_error':
            try:
                self.rf_seds_err
            except AttributeError:
                raise Exception("Error not computed when shifting to rest frame. "
                                "Please run to_rest_frame() with compute_error=True or "
                                "change the parameter 'weight' to None")

            use_errors = True

        # Parsing the bin_dict
        bin_dict_old = bin_dict.copy()
        bin_dict = {}
        discrete_bins = {}
        for key, bin_edgs in bin_dict_old.items():
            if key[-2:] != '==':
                if not isinstance(bin_edgs[0], (tuple, list)):
                    bin_edgs = [bin_edgs]

                bin_edgs_new = list()
                for bin_edg in (bin_edgs):
                    for i in range(len(bin_edg) - 1):
                        bin_edgs_new.append([bin_edg[i], bin_edg[i+1]])

                bin_edgs = bin_edgs_new
                discrete_bin = False

            else:
                discrete_bin = True

            if key[-2:] == '%%':
                bin_edgs = [np.percentile(self.df[key[:-2]], bin_edg)
                            for bin_edg in bin_edgs]

            if key[-2:] == '%%' or key[-2:] == '==':
                key = key[:-2]

            bin_dict[key] = bin_edgs
            discrete_bins[key] = discrete_bin

        # Cropping wl_grid and rf_seds if the redshift binning is more restrictive
        if self.z_label in bin_dict.keys():
            wl_ind_min = np.argmin(np.abs(self.wl_grid - self.wl_obs_min /
                                          (1 + bin_dict[self.z_label][-1][-1])))
            wl_ind_max = np.argmin(np.abs(self.wl_grid - self.wl_obs_max /
                                          (1 + bin_dict[self.z_label][0][0])))
            self.wl_grid = self.wl_grid[wl_ind_min:wl_ind_max+1]
            self.rf_seds = self.rf_seds[:, wl_ind_min:wl_ind_max+1]
            if use_errors:
                self.rf_seds_err = self.rf_seds_err[:, wl_ind_min:wl_ind_max+1]

        # Computing weights
        if not weight:
            weights = np.ones(self.rf_seds.shape)

        elif weight.lower() == 'inv_variance':
            weights = self.rf_seds_err**-2

        elif weight.lower() == 'snr_square':
            weights = (self.rf_seds / self.rf_seds_err)**2

        else:
            raise ValueError(f"{weight} is not a valid weight type."
                             " Please select 'inv_variance', 'snr_square' or None")

        # Generating the empty xarray of stacked seds
        bin_mid_dict = {}
        for key, bin_edgs in bin_dict.items():
            if discrete_bins[key]:
                bin_mid_dict[key] = bin_edgs

            else:
                bin_mid = [np.mean(bin_edg) for bin_edg in bin_edgs]
                bin_mid_dict[key] = bin_mid

        stacks_shape = [len(bin_edg) for bin_edg in bin_dict.values()]
        stacks_shape += [3, len(self.wl_grid)]
        stacks_dims = list(bin_dict.keys()) + ['data', 'rf_wl']
        stacks_coords = bin_mid_dict.copy()
        stacks_coords['data'] = ['flux', 'flux_error', 'counts']
        stacks_coords['rf_wl'] = self.wl_grid

        attr_dict = {'flux_units': f'{self.flux_units}',
                     'flux_units_latex': f'{self.flux_units:latex_inline}',
                     'wavelength_units': f'{self.wavelength_units}',
                     'wavelength_units_latex': f'{self.wavelength_units:latex_inline}',
                     'flux_conversion': self.flux_conversion,
                     'z_label': self.z_label, 'min_n_obj': min_n_obj}
        if weight:
            attr_dict['weight'] = weight
        else:
            attr_dict['weight'] = 'None'

        if error_type:
            attr_dict['error_type'] = error_type
        else:
            attr_dict['error_type'] = 'None'

        for key, bin_edgs in bin_dict.items():
            if not discrete_bins[key]:
                bin_edg_min = [bin_edg[0] for bin_edg in bin_edgs]
                bin_edg_max = [bin_edg[1] for bin_edg in bin_edgs]
                attr_dict[f'{key}_min'] = bin_edg_min
                attr_dict[f'{key}_max'] = bin_edg_max

        stacked_seds = xr.DataArray(np.zeros(stacks_shape), dims=stacks_dims,
                                    coords=stacks_coords, attrs=attr_dict)

        # Iterating over all cases and stacking
        it = np.nditer(stacked_seds[..., 0, 0], flags=['multi_index'])
        bin_labels = list(bin_dict.keys())
        bin_edgs = list(bin_dict.values())
        select_z = ((self.df[self.z_label].values >= self.z_min) 
                    & (self.df[self.z_label].values <= self.z_max))

        df_tmp = self.df[select_z]
        for x in it:
            # Selecting objects for bin
            select = np.ones(len(df_tmp)).astype(bool)
            for i, item in enumerate(it.multi_index):
                bin_edg = bin_edgs[i][item]
                if discrete_bins[bin_labels[i]]:
                    select *= df_tmp[bin_labels[i]] == bin_edg
                else:
                    select *= (df_tmp[bin_labels[i]] >= bin_edg[0]
                               ) & (df_tmp[bin_labels[i]] < bin_edg[1])

            rf_seds_tmp = self.rf_seds[select]
            weights_tmp = weights[select]
            if use_errors:
                rf_seds_err_tmp = self.rf_seds_err[select]

            # Computing stacked flux, error and counts
            stack_sed = np.ma.average(rf_seds_tmp, weights=weights_tmp, axis=0)
            stack_counts = np.sum(~rf_seds_tmp.mask, axis=0)
            stack_sed.mask += stack_counts < min_n_obj
            if error_type:
                if error_type.lower() == 'flux_error':
                    stack_sed_err = np.ma.sqrt(np.ma.sum(rf_seds_err_tmp**2 * weights_tmp**2, axis=0)
                                               / np.ma.sum(weights_tmp, axis=0)**2)

                elif error_type.lower() == 'std' or error_type.lower() == 'std_mean':
                    variance = np.ma.average(
                        (rf_seds_tmp - stack_sed)**2, weights=weights_tmp, axis=0)
                    stack_sed_err = np.ma.sqrt(variance)

                    if error_type.lower() == 'std_mean':
                        stack_sed_err /= np.sqrt(stack_counts)

            else:
                stack_sed_err = np.full(self.wl_grid.shape[0], np.nan)

            # renormalizing if the rest-frame shift was normalized. Just in case
            if self.flux_conversion == 'normalized':
                #normalization = np.trapz(stack_sed, self.wl_grid[~stack_sed.mask])
                normalization = np.trapz(stack_sed, self.wl_grid)
                wl_max = self.wl_grid[~stack_sed.mask][-1]
                wl_min = self.wl_grid[~stack_sed.mask][0]
                stack_sed = stack_sed / normalization * (wl_max - wl_min)
                stack_sed_err = stack_sed_err / \
                    normalization * (wl_max - wl_min)

            # Storing
            self.stacked_seds = stacked_seds
            self.stack_sed = stack_sed
            self.stack_sed_err = stack_sed_err
            self.stack_counts = stack_counts
            stacked_seds[it.multi_index + (0,)] = stack_sed
            stacked_seds[it.multi_index + (1,)] = stack_sed_err
            stacked_seds[it.multi_index + (2,)] = stack_counts

        self.stacked_seds = stacked_seds
        self.stack_saved = False

    def plot(self, line_label=None, column_label=None, row_label=None,
             counts=False, spectral_lines=False, logscale=False,
             wavelength_min=None, wavelength_max=None,
             aspect_ratio=1.62, fig_title=False, show=True, rc_params=None):
        '''
        Plots the stacked SEDs from the `stacked_seds` xarray.

        Plots the stacked SEDs from the `stacked_seds` xarray. All stacking 
        bins will be plotted; the number of files generated will depend on 
        the parameters. Filenames will be automatically generated.

        If some labels or titles overlap or are cropped in the saved plots, 
        testing different `aspect_ratio` values may solve it.

        Parameters
        ----------
        line_label : str, optional
            Label of the binning that should be plotted as different lines in 
            each axis. The default is None.
        column_label : str, optional
            Label of the binning that should be displayed as different subplot 
            columns in the figure. If specified and `row_label` = None, it will be
            be displayed in all subplots instead. The default is None.
        row_label : str, optional
            Label of the binning that should be displayed as different subplot 
            rows in the figure. If specified and `column_label` = None, it will be
            be displayed in all subplots instead. The default is None.
        counts : bool, optional
            Display in the y-axis the number of galaxies stacked per wavelength 
            grid point instead of the stacked fluxes. The default is False.
        spectral_lines : bool or dict, optional
            If bool, decides if spectral lines must be marked or not in the
            plots. If dict, marks the spectral lines, but the provided dict
            values, instead of the default ones. (OII, OIII, Halpha, Hbeta).
            The dictionary keys must have LaTeX string format.
            The default is False.
        logscale: bool, optional
            Display the y axes in logarithmic scale. Will raise an error if 
            anything in the y axis has negative values. The default is False.
        wavelength_min: float, optional
            Minimum wavelength to be plotted. Does not simply adjust the plot 
            limits; the stacked SEDs will be cropped to wavelength_min, and then
            plotted. The default is None.
        wavelength_max: float, optional
            Maximum wavelength to be plotted. Does not simply adjust the plot 
            limits; the stacked SEDs will be cropped to wavelength_max, and then
            plotted. The default is None.
        aspect_ratio : float, optional
            Aspect ratio of the subplots. The default is 1.62.
        fig_title : bool, optional
            Add an automatically generated title to the figure, specifying all 
            the bins that are not in line_label, column_label and/or row_label.
            The default is False.
        show : bool, optional
            Show the plots once generated. The default is True.
        rc_params : dict, optional
            Dictionary of parameters to pass to `matplotlib.rcParams`.
            Use it to modify the default plotting layout. The default is None.
        Returns
        -------
        None.

        '''

        try:
            self.stacked_seds
        except AttributeError:
            raise Exception("No stacked SEDs available. Please stack a loaded"
                            " catalog with stack() or load an xarray"
                            " of stacked SEDs with load_stack()")

        if not self.stack_saved:
            raise Exception("The stack has not been saved. Please run"
                            " save_stack() before plotting")

        if spectral_lines:
            if isinstance(spectral_lines, dict):
                spectral_lines_dict = spectral_lines
            else:
                spectral_lines_dict = {'OII': [372.68], 'OIII': [495.9, 500.7],
                                       'H$\\alpha$': [656.28], 'H$\\beta$': [486.1],
                                       'MgII': [280]}

        self._rc_parameters(rc_params=rc_params)

        # Sorting out columns/rows, labels and format
        xlabel = rf"$\lambda$ ({self.stacked_seds.attrs['wavelength_units_latex']})"
        if counts:
            ylabel = 'N obj'
            
        elif (self.stacked_seds.attrs['flux_conversion'] == 'normalized'
              or self.stacked_seds.attrs['flux_conversion'] == 'redshift_normalized'):
            ylabel = 'Normalized flux'
            
        elif self.stacked_seds.attrs['flux_conversion'] == 'luminosity':
            if self.flux_density == 'wavelength':
                ylabel = rf"$L_\lambda$ ({self.stacked_seds.attrs['flux_units_latex']})"                
            elif self.flux_density == 'frequency':
                ylabel = rf"$L_\nu$ ({self.stacked_seds.attrs['flux_units_latex']})"
                
        else:
            if self.flux_density == 'wavelength':
                ylabel = rf"$f_\lambda$ ({self.stacked_seds.attrs['flux_units_latex']})"
            elif self.flux_density == 'frequency':
                ylabel = rf"$f_\nu$ ({self.stacked_seds.attrs['flux_units_latex']})"

        fig_labels = list(self.stacked_seds.dims)
        fig_labels.remove('data')
        fig_labels.remove('rf_wl')
        if line_label:
            fig_labels.remove(line_label)

        if column_label:
            fig_labels.remove(column_label)

        if row_label:
            fig_labels.remove(row_label)

        kw = {column_label: 0, line_label: 0, row_label: 0}
        try:
            kw.pop(None)
        except:
            pass

        subplot_label = None
        if column_label and not row_label:
            subplot_label = column_label
        elif row_label and not column_label:
            subplot_label = row_label

        if line_label:
            n_lines = self.stacked_seds[line_label].shape[0]
            legend_labels = [self._range_label(
                line_label, i) for i in range(n_lines)]
        else:
            n_lines = 1
            legend_labels = None

        if not column_label and not row_label:
            n_cols = 1
            n_rows = 1
        elif subplot_label:
            n_cols, n_rows = self._determine_cols_rows(
                self.stacked_seds[subplot_label].shape[0], 1)
        else:
            n_cols = self.stacked_seds[column_label].shape[0]
            n_rows = self.stacked_seds[row_label].shape[0]

        if n_cols == 1 and n_rows == 1:
            fontsize = 'large'
        elif max(n_cols, n_rows) < 3:
            fontsize = 'x-large'
        else:
            fontsize = 'xx-large'

        plt.rcParams.update({'axes.labelsize': fontsize, 'axes.titlesize': fontsize,
                            'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize, })

        # Applying wavelength ranges if specified
        wl_grid = self.stacked_seds['rf_wl'].data
        if wavelength_min:
            ind_min = np.argmin(np.abs(wl_grid - wavelength_min))
        else:
            ind_min = None

        if wavelength_max:
            ind_max = np.argmin(np.abs(wl_grid - wavelength_max))
        else:
            ind_max = None

        stacked_seds_cropped = self.stacked_seds[..., ind_min:ind_max]

        # Iterating over all necessary subplots
        it = np.nditer(stacked_seds_cropped.isel(
            **kw, data=0, rf_wl=0), flags=['multi_index'])
        for _ in it:
            inds = it.multi_index
            kw = {label: inds[i] for i, label in enumerate(fig_labels)}
            stacked_seds_tmp = stacked_seds_cropped.isel(**kw)
            if self.stacked_seds.attrs['flux_conversion'] == 'normalized':
                sharey = True
            else:
                sharey = False

            filename = [f'{label}{inds[i]}' for i,
                        label in enumerate(fig_labels)]
            if subplot_label:
                filename.append(f's-{subplot_label}')
            else:
                if row_label:
                    filename.append(f'r-{row_label}')
                if column_label:
                    filename.append(f'c-{column_label}')

            if line_label:
                filename.append(f'l-{line_label}')

            if counts:
                filename.append('counts')

            filename = '_'.join(filename)
            if filename == '':
                filename = 'full_sample'

            gridspec_dict = {}
            if subplot_label:
                gridspec_dict['hspace'] = 0.15
            else:
                gridspec_dict['hspace'] = 0.05

            if sharey:
                gridspec_dict['wspace'] = 0.05 / aspect_ratio
            else:
                gridspec_dict['wspace'] = 0.3 / aspect_ratio

            fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, dpi=160, sharex=True, sharey=sharey,
                                   figsize=[6.4 * n_cols, 6.4 *
                                            n_rows / aspect_ratio],
                                   gridspec_kw=gridspec_dict)

            if not isinstance(ax, np.ndarray):
                ax = np.array([ax])

            if fig_title:
                suptitle = [self._range_label(label, inds[i])
                            for i, label in enumerate(fig_labels)]
                suptitle = ', '.join(suptitle)
                if suptitle != '':
                    plt.suptitle(suptitle, wrap=True,
                                 fontsize=fontsize, y=0.95)

            it_ax = np.nditer(ax, flags=['multi_index', 'refs_ok'])
            for _ in it_ax:
                inds = it_ax.multi_index
                kw = {}
                if n_cols == 1:
                    i = inds[0]
                    j = 0
                elif n_rows == 1:
                    i = 0
                    j = inds[0]

                else:
                    i, j = inds

                kw_plot = {'line_label': line_label, 'counts': counts,
                           'logscale': logscale, 'sharey': sharey}
                if spectral_lines:
                    kw_plot['spectral_lines_dict'] = spectral_lines_dict

                if subplot_label:
                    index = i*n_cols + j
                    kw = {subplot_label: index}
                    if index >= stacked_seds_tmp[subplot_label].shape[0]:
                        break

                    title = self._range_label(subplot_label, index)
                    kw_plot['title'] = title

                elif column_label and row_label:
                    kw = {row_label: i, column_label: j}
                    extra_xlabel = self._range_label(column_label, j)
                    extra_ylabel = self._range_label(row_label, i)
                    if i == 0:
                        kw_plot['extra_xlabel'] = extra_xlabel
                    if j == n_cols - 1:
                        kw_plot['extra_ylabel'] = extra_ylabel

                if i == n_rows - 1:
                    kw_plot['xlabel'] = self._get_alias(xlabel)

                if j == 0:
                    kw_plot['ylabel'] = self._get_alias(ylabel)

                if i == 0 and j == 0:
                    kw_plot['legend_labels'] = legend_labels
                if i == n_rows - 1 and j == n_cols - 1:
                    kw_plot['spectral_lines_legend'] = True

                self._single_plotter(ax[inds], stacked_seds_tmp, kw, **kw_plot)

                # if self.stacked_seds.attrs['flux_units'].lower() == 'normalized':
                #    ax[0,0].set_ylim(bottom=max(1e-5, ax[0,0].get_ylim()[0]))

            if fig_title:
                plt.savefig(f'{self.stack_folder}{filename}.pdf',
                            transparent=True)
                plt.savefig(f'{self.stack_folder}{filename}.png',
                            transparent=True)

            else:
                plt.savefig(f'{self.stack_folder}{filename}.pdf',
                            bbox_inches='tight', transparent=True)
                plt.savefig(f'{self.stack_folder}{filename}.png',
                            bbox_inches='tight', transparent=True)

            if show:
                plt.show()
            else:
                plt.close()


if __name__ == '__main__':
    master_cat_dir = '/home/pablo/observational_data/PAUS_master_catalog/PAUS_master_catalog_zw_fluxes_only_unmasked_galaxies.parquet'
    st = Stacker()
    st.load_catalog(master_cat_dir, z_label='zw')
    st.to_rest_frame(flux_conversion='redshift_normalized', z_min=0.1, z_max=1.5,
                     show_lum_plot=True)
