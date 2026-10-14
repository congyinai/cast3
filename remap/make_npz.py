"""Convert an ESMF offline regridding weight file to the compact .npz used by this repository.

    python remap/make_npz.py esmf_ERA5_to_C192_r2_China_conserve.nc remap/era5_to_c192_conserve.npz
    python remap/make_npz.py esmf_C192_r2_China_to_ERA5_conserve.nc remap/c192_to_era5_conserve.npz

Only the sparse matrix is kept: weights S (float32) and 0-based target/source indices row, col (int32).
Grid coordinates, cell areas and the other fields of the ESMF file are not needed for regridding.
"""
import sys

import numpy as np
import xarray as xr

src, dst = sys.argv[1], sys.argv[2]
with xr.open_dataset(src) as w:
    np.savez_compressed(
        dst,
        S=w["S"].values.astype(np.float32),
        row=(w["row"].values - 1).astype(np.int32),     # ESMF indices are 1-based
        col=(w["col"].values - 1).astype(np.int32),
    )
print("saved", dst)
