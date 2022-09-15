# stonp (STacker Of Narrow-band Photometry)

stonp reads photometric galaxy catalogs, shifts them to rest frame and stacks the data in order to obtain average spectral energy distributions (SEDs).

Just run
```
from stonp import Stacker

st = Stacker()
st.load_catalog(...)
st.to_rest_frame(...)
st.stack(...)
st.plot(...)
```

and get publication-ready SED plots from your data!


## Features
- Works with any arbitrary set of photometric bands
- Several options for flux conversion, error calculation and weighting
- Splits the catalog in any multi-dimensional binning to derive SEDs for different populations and/or measurements
- Saves the stacked SEDs in a multi-dimensional xarray with all necessary metadata
- Plots the results with a highly versatile API


## Installation

Clone the repository, move into its root and run

```
pip install -e .
```

## Documentation
https://stonp.readthedocs.io 
Still working on it!

## Citations
coming soon!

## License
The project is licensed under the GNU General Public License v3.0
