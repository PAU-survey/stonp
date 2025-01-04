.. _quickstart:

Getting started
===============


stonp is a module that loads photometric galaxy catalogs and stacks their data to obtain average SEDs, which are stored in a :ref:`stacked_seds <stacked_seds>` array. This is all done through a single object, the :ref:`Stacker <stacker>` class Each `Stacker` instance may contain a single photometric catalog, and produces a single `stacked_seds` array. Therefore, in order to work simultaneously with different catalogs or different stacking configurations, different `Stacker` instances must be created.


To import stonp and create a new `Stacker` instance, just type::

    import stonp
    st = stonp.Stacker()


From here, we will see the basic steps of the stonp workflow. For a detailed description of all methods and its parameters, please check :ref:`the API of the Stacker class <stacker>`.


1) Loading a catalog
--------------------

You can load a photometric galaxy catalog as follows::

    st.load_catalog('path/to/catalog.csv')


This method will load the catalog into the `Stacker` instance, performing some basic data cleaning in the process (removing objects with too many missing band fluxes, interpolating band fluxes if necessary). The requisites for the catalog file are:

* A .csv files with no other header than the column labels.
* Band fluxes must be in units of spectral wavelength density:

.. math:: f_{\lambda}s

* One single row per object is allowed.
* At least one column must contain redshift data. Objects with invalid redshift data will be dismissed.
* If present, the band flux errors must be labelled as `band_flux_label` + `error_suffix`.

When running `st.load_catalog()`, some more information regarding the catalog data can be specified, such as:

* Maximum number of acceptable invalid band fluxes (`max_nan_bands`). 
* Redshift and flux error labels (`z_label`, `bands_error_suffix`).
* Band wavelength and flux units (`wavelength_units`, `flux_units`).
* Band response functions (`bands_data`, either as a .json file or as a dict).

If any of these need to be changed, `st.load_catalog()` must be run again.


2) Shifting to rest frame
-------------------------

Once the catalog data is loaded, the next step is to shift the band fluxes of all objects to rest frame::

    st.to_rest_frame()


The shifting to rest frame is performed in two steps:

1) Calculation of a common rest-frame wavelength grid, based on the catalog data and input parameters of `st.to_rest_frame()`.
2) Interpolation of all the band flux data to the rest-frame grid, following:

.. math::

    \lambda_{rest} = \lambda_{obs} / (1 + z); \quad f_{\lambda\, rest} = f_{\lambda\, obs} \cdot (1 + z)


Several options can be specified when shifting to rest frame, namely:

* Flux unit conversion: Any conversion to the band flux units is performed now (`flux_conversion`), right after shifting to rest frame. You can leave the fluxes as they are (`flux_conversion` = 'nothing'), or convert to the following units:
    + Luminosity wavelength density (`flux_conversion` = 'luminosity'): Fluxes converted to luminosities. Recommended when stacking in bins on very similar luminosities, or very low SNR measurements where individual detections are not possible.

    .. math:: L_{\lambda} = 4\pi D_L(z)^2 f_{\lambda}

    + Normalized flux (`flux_conversion` = 'normalized'): The norm of all SEDs is equal to their wavelength span (in rest frame). Recommended when stacking objects in a wide luminosity range.

    .. math:: 

        f_{\lambda\, norm} = \frac{f_{\lambda}}{\lambda_{max} - \lambda_{min}} / \int_{\lambda_{min}}^{\lambda_{max}} f_{\lambda}(\lambda) d\lambda

* Interpolation type (`use_band_responses`): Two different methods can be used to interpolate the band fluxes to the rest-frame grid. A linear interpolation of the closest bands (`use_band_responses` = False) can be used, or a weighted average of all bands (`use_band_responses` = True), where the weights for a given wavelength grid point lambda and band n are computed as shown below. This weighting requires the band response functions to have been loaded as a .json file, but it is more accurate than linear interpolation, and specially recommended if an inhomogeneous band set is being used (e.g., mixed broad and narrow bands, or narrow bands of irregular coverage).

.. math::

    w_{n\, \lambda} = \frac{\int^{\lambda + \Delta \lambda / 2}_{\lambda - \Delta \lambda / 2} R_n(\lambda \cdot (1+z)) d\lambda}{\int^\infty_0 R_n(\lambda \cdot (1 +z))d\lambda}; \quad f_\lambda = \frac{\sum_n w_{n\, \lambda}^2 f_{n\, \lambda}}{\sum_n w_{n\, \lambda}^2}

* Error computation (`compute_error`): The flux errors may or may not be propagated when shifting to rest frame. Not doing so will significantly speed up the process, but will not allow to use flux errors in any of the subsequent steps.

* Several wavelength grid options (`wl_rf_step`, `wl_obs_min`, `wl_obs_max`, `z_min`, `z_max`).


3) Stacking
-----------

When the `Stacker` instance has all fluxes shifted to a common rest-frame wavelength grid, we can stack the fluxes to obtain average SEDs::

    st.stack()


This will compute the average rest-frame SEDs in all of the specified bins, and store them in a :ref:`stacked_seds <stacked_seds>` array. The following options can be specified:

* Binning in which the objects must be stacked, specified as a dictionary (`bin_dict`). Please refer to the :ref:`bin_dict page<bin_dict>` for a complete explanation of how to specify the binning.
* Weighting (`weight`): By default, for each bin the unweighted average SED will be computed. However, a inverse variance weighting (`weight` = 'inv_variance') can be applied, or SNR squared weighting (`weight` = 'snr_square'). Both options require to have propagated the error when shifting to rest frame (`st.to_rest_frame(compute_error=True)`).
* The error of the stacked SEDs (`error_type`). By default, no error will be computed for the stacked fluxes, but you you can specify the error to be the propagated flux error of the average (`error_type` = 'flux_error'), the standard deviation of all the stacked fluxes (i.e., sample variance, `error_type` = 'std'), or the standard deviation of the mean (`error_type` = 'std_mean').
* Minimum number of objects per wavelength grid point (`min_n_obj`). If a given wavelength point has less objects than `min_n_obj`, its stacked flux and error will be set to NaN.


4) Saving the stack
-------------------

Finally, the `stacked_seds` array that has been generated can be saved into a given directory::

    st.save_stack('path/to/stack_folder')

The `stacked_seds` array will be saved inside this directory as a netCDF file (stacked_seds.nc), which can be read and manipulated with `Xarray <https://xarray.dev/>`_. In addition to saving the array, you can also make the `Stacker` instance return the `stacked_seds` array it is currently working with, to examine it directly. Just type::

    stacked_seds = st.return_stack()


Loading a stack
----------------

So fa, we have seen all the steps necessary to produce a `stacked_seds` array from a photometric galaxy catalog. However, if you just want to plot an already existing stack, you can load it as follows::

    st.load_stack('path/to/stack_folder')

The `stacked_seds` array is expected to be inside the specified stack folder, with the name stacked_seds.nc. If a `stacked_seds` array was already computed with a given `Stacker` instance (`st` in this example), loading another `stacked_seds` array will overwrite the current array from the instance namespace, so be sure to save it first.
 


5) Plotting
-----------

With an `stacked_seds` array already computed (either by following steps 1) to 4), or simply loading an already existing one), we can plot the stacked SEDs::

    st.plot()

All plots generated through this method will be saved as both .png and .pdf files in the same 'stack_folder' specified when running `st.save_stack()` or `st.load_stack()`. Therefore, all plot files are stored in the same directory as the stacked_seds.nc file they came from.

The `st.plot()` method will automatically generate plot files for all possible bin combinations of the `stacked_seds` array. However, each plot file may be a grid with a given number of subplots, and each subplot may contain several stacked SEDs. This is controlled with the following parameters:

* `line_label`: The label of the binned quantity to be displayed as different SEDs in each subplot.
* `column_label`: The label of the binned quantity to be displayed as columns in the subplot grid.
* `row_label`: The label of the binned quantity to be displayed as rows in the subplot grid.

All these labels must be specified as they appear in the photometric galaxy catalog. If `row_label` (`column_label`) is specified without specifying `column_label` (`row_label`), an approximately square grid of subplots will be generated, not a grid with a single row (column). 

The filenames for all the plot files will be automatically generated according to the `line_label`, `column_label` and `row_label` parameters. For a detailed list of all the other plotting options, please check :ref:`the Stacker class API <stacker>`.





