.. _bin_dict:

The binning dictionary (bin_dict)
=================================

The most important parameter to specify when stacking a galaxy catalog is the binning; i.e., the cuts that must be applied to the catalog before stacking, in order to obtain average SEDs for different subsamples. This allows to study how the average SED changes according to different galaxy properties (e.g., redshift, stellar mass, color, morphology, etc). stonp allows to specify any arbitrary binning of any given number of properties, as long as all of them are columns of the loaded galaxy catalog.

The binning is specified with the `bin_dict` parameter of the `st.stack()` method (check the :ref:`Getting started <quickstart>` section for an overview). This `bin_dict` parameter must be a dictionary, with the following general syntax::

    st.stack(bin_dict={'label1' : [binning1], 'label2' : [binning2], ...})


For each item of the dictionary, the key must be the label of the column in the galaxy catalog we want to apply the binning to (plus some optional suffixes). The values of each item must be a list specifying limits of each bin. Two different kinds of binning may be performed: continuous and discrete binning. 


Continuous binning
------------------


In continuous binning, the property we are binning has a continuous range of values, and thus each bin comprises all objects with the aforementioned value within the bin edges. To specify a continuous binning, just use as a key the column label as it is, and specify the binning as a list of lists. Each one of the nested lists must have two elements: the lower and upper edges of the respective bin. These edges must be either floats o ints.
For example, assuming a redshift column labeled 'zb', we can specify the following `bin_dict`::

    bin_dict = {'zb' : [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8]]}


This will produce three different redshift between zb=0.5 and zb=0.8, each one of width Δz=0.1. The lower bound of each bin will be closed, and the upper bound open (e.g., the first bin in our example will be 0.5 <= 'zb' < 0.6). This open/closed bound criterion can not be modified.

For a set of contiguous bins, instead of specifying the bounds of each bin in a separate nested list, you can just specify the edges of each bin in the same list. For example::

    bin_dict = {'zb' : [0.5, 0.6, 0.7, 0.8]}


will produce the exact same binning as the previous case. Both the syntax for the separate and contiguous bins can be mixed, e.g::

    bin_dict = {'zb' : [[0.5, 0.6, 0.7, 0.8], [1, 1.2]]}


will produce the same binning as::

    bin_dict = {'zb' : [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [1, 1.2]]}


In these last two examples, all galaxies with 0.8 <= 'zb' < 1 will not be included in any bin, and thus excluded from stacking.


Binning with percentiles
++++++++++++++++++++++++

Continuous binning can also be specified as percentiles. To do so, just append '%%' to the column label used as a key in 'bin_dict'. This suffix just tells stonp to use percentiles, and will be removed before querying the galaxy catalog. Therefore, no modifications to the actual galaxy catalog are needed. 

For example, let us assume that our galaxy catalog also has a column for the logarithm of stellar mass: 'sm_log'. If we want to apply a uniform redshift binning similar to before, and a stellar mass binning divided in the four quartiles, we should input::

    bin_dict = {'zb' : [0.5, 0.6, 0.7, 0.8], 'sm_log%%' : [0, 25, 50, 75, 100]}


The syntax for specifying the bin edges (separate or contiguous) works exactly the same; the only difference is that the bin edges provided in the list are interpreted as percentiles. Lower bounds will be closed and open bounds open as in the previous cases.



Discrete binning
----------------

Discrete binning allows to bin properties that have discrete values, such as galaxy types, AGN classifications, or any flags derived from data reduction. To apply a discrete binning, append '==' to the column label used as a key. The bin values must be specified as a list; since there are no bin upper and lower bounds (given that we are checking for equality), no nested lists are needed.

For example, let us assume that our catalog also has an AGN classification; the column is labeled 'agn', and the values are boolean (True or False). If we want to use the previous binning, but also separate between galaxies with and without AGNs, we have to input::

    bin_dict = {'zb' : [0.5, 0.6, 0.7, 0.8], 'sm_log%%' : [0, 25, 50, 75, 100], 'agn==' : [True, False]}


The bin values for discrete binning may be any data type, but since stonp checks for exact equality, we advise against using floats.

