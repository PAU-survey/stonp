#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

import numpy as np
import pandas as pd

from scipy import interpolate


def linterp(x, xp, yp, yp_err):
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

    if not isinstance(x, (np.ndarray, list)):
        raise TypeError('x must be a numpy array or a list')
    if not isinstance(xp, (np.ndarray, list)):
        raise TypeError('xp must be a numpy array or a list')
    if not isinstance(yp, (np.ndarray, list)):
        raise TypeError('yp must be a numpy array or a list')
    if not isinstance(yp_err, (np.ndarray, list)):
        raise TypeError('yp_err must be a numpy array or a list')
    if not len(xp) == len(yp) == len(yp_err):
        raise ValueError('xp, yp, and yp_err must have the same length')

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


def json_loader(bands_data_dir, df=None, sort=True):
    # Loads the .json of band response functions specified
    # Returns band labels, average wavelengths, and interpolated response
    # functions
    if not isinstance(bands_data_dir, str):
        raise TypeError('bands_data_dir must be a string')
    if not isinstance(df, (pd.DataFrame, type(None))):
        raise TypeError('df must be a pandas DataFrame or None')
    if not isinstance(sort, bool):
        raise TypeError('sort must be a boolean')
    with open(bands_data_dir, 'r', encoding='utf-8') as read_file:
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
        band_mean_wls[key] = np.trapezoid(wl * r, wl) / np.trapezoid(r, wl)

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
    wl_grid_obs = np.arange(wl_bands_min / wl_bands_step,
                            wl_bands_max / wl_bands_step + 1) * wl_bands_step

    # Interpolating to a single array the band responses
    r_nb = np.zeros([len(band_responses), wl_grid_obs.shape[0]])
    i = 0
    for value in band_responses.values():
        r_nb[i, :] = np.interp(wl_grid_obs, value['wavelength'], value['response'],
                               left=0, right=0)
        r_nb[i, :] /= np.trapezoid(r_nb[i, :], wl_grid_obs)
        i += 1

    # Computing the interpolation object
    r_nb = interpolate.interp1d(
        wl_grid_obs, r_nb, bounds_error=False, fill_value=0)

    nb_labels = list(band_mean_wls.keys())
    wl_nb = np.array(list(band_mean_wls.values()))

    # Maybe we could include something to crop unnecessary wavelengths
    # in wl_grid_obs and save memory when computing smoothing band
    return nb_labels, wl_nb, r_nb, wl_grid_obs


def bin_dict_parser(bin_dict):
    # Parses the bin_dict so all bins are nested lists of two elements

    if not isinstance(bin_dict, dict):
        raise TypeError('bin_dict must be a dictionary')

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


def determine_cols_rows(n_subplots, aspect_ratio):
    # Determines the number of columns a rows for a plot with a given
    # number of subplots and aspect ratio (assuming all subplots square)

    n_cols = int(np.sqrt(n_subplots) * aspect_ratio)
    n_rows = int(np.sqrt(n_subplots) / aspect_ratio)
    while n_rows * n_cols < n_subplots:
        if n_cols <= n_rows:
            n_cols += 1
        else:
            n_rows += 1

    return n_cols, n_rows


def query_yes_no(question, default="yes"):
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
        if choice in valid:
            return valid[choice]
        # else:
        sys.stdout.write(
            "Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
