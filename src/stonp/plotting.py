#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def single_plotter(ax, stacked_seds_tmp, kw, line_label=None,
                   xlabel=None, ylabel=None, legend_labels=None, title=None,
                   extra_xlabel=None, extra_ylabel=None, counts=False,
                   spectral_lines_dict=None, spectral_lines_legend=False,
                   logscale=False, sharey=False):
    """
    Parameters
    ----------
    ax : array_like
    stacked_seds_tmp : dict
    kw : dict
    line_label : , optional
    xlabel : , optional
    ylabel : , optional
    legend_labels : , optional
    title : , optional
    extra_xlabel : , optional
    extra_ylabel : , optional
    counts : , optional
    spectral_lines_dict : , optional
    spectral_lines_legend : , optional
    logscale : bool, optional
    sharey : bool, optional

    Returns
    -------
    None
    """
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


def rc_parameters(rc_params=None):
    # Defines the default parameters for plotting layout with matplotlib
    if not isinstance(rc_params, dict) and not rc_params is None:
        raise TypeError('rc_params must be a dictionary')

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

    return plt.rcParams
