"""Build a CAST3 C192 initial condition from ERA5.

Usage
-----
    python prepare_init.py --time 2025-04-01T00 --out data/2025040100.nc --download
    python prepare_init.py --time 2025-04-01T00 --out data/2025040100.nc  era5_pressure.nc era5_surface.nc

Input
-----
With --download, ERA5 is read from ARCO-ERA5, the public copy of ERA5 on Google Cloud Storage
(https://github.com/google-research/arco-era5). No account is needed, and only the requested time is
downloaded (about 1 GB, as each chunk holds all 37 pressure levels). ERA5T, the preliminary ERA5,
is available there about one week after real time.

Otherwise: one or more NetCDF or GRIB files on the 0.25 degree ERA5 grid (721 x 1440) that together contain

    pressure levels (at least the 13 levels in common.PRESSURE_LEVELS):
        geopotential (z), specific_humidity (q), temperature (t),
        u_component_of_wind (u), v_component_of_wind (v)
    surface:
        mean_sea_level_pressure (msl), 10m_u_component_of_wind (u10),
        10m_v_component_of_wind (v10), 2m_temperature (t2m)

Both the long names and the ERA5 short names (in brackets) are accepted. Other variables, levels and
times in the files are ignored. Files downloaded from the Copernicus Climate Data Store work as they are.

Output
------
One NetCDF file on the C192 grid, the input format of infer.py: the nine variables conservatively
regridded to the cubed sphere, plus the cosine of the solar zenith angle at the initial time.
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
from earthkit.meteo.solar import cos_solar_zenith_angle

from common import (C192_DIMS, C192_SHAPE, LATLON_SHAPE, PRESSURE_LEVELS, PRESSURE_VARS,
                    SURFACE_VARS, UNITS, Regridder)

ERA5_SHORT_NAMES = {
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
}
ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# names used for the same coordinate by different ERA5 sources (CDS NetCDF, GRIB via cfgrib, ...)
COORD_ALIASES = {
    "valid_time": "time",
    "pressure_level": "level",
    "isobaricInhPa": "level",
    "lat": "latitude",
    "lon": "longitude",
}


def open_era5(path):
    """Open one ERA5 file and rename variables and coordinates to the names used here."""
    if path.endswith((".grib", ".grb", ".grib2")):
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    else:
        ds = xr.open_dataset(path)

    if "valid_time" in ds.coords and "time" in ds.dims:
        ds = ds.drop_vars("valid_time")         # GRIB analyses: valid_time == time
    renames = {old: new for old, new in {**ERA5_SHORT_NAMES, **COORD_ALIASES}.items()
               if old in ds.variables}
    return ds.rename(renames)


def open_arco_era5():
    """Open ARCO-ERA5 on Google Cloud. Lazy: data are only downloaded when a field is read."""
    return xr.open_zarr(ARCO_ERA5, chunks=None, storage_options={"token": "anon"})


def read_field(datasets, name, init_time):
    """Find variable `name` in the datasets and return it at `init_time` on the ERA5 grid.

    Returns (13, 721, 1440) for pressure-level variables and (721, 1440) for surface variables,
    latitude north to south and longitude 0 ... 359.75.
    """
    for ds in datasets:
        if name in ds:
            field = ds[name]
            break
    else:
        raise KeyError(f"{name} not found in the input files")

    if "time" in field.dims:
        field = field.sel(time=init_time)
    elif "time" in field.coords and pd.Timestamp(field["time"].values) != init_time:
        raise ValueError(f"{name}: file time {field['time'].values} is not {init_time}")
    if name in PRESSURE_VARS:
        field = field.sel(level=PRESSURE_LEVELS)

    field = field.sortby("latitude", ascending=False)
    field = field.assign_coords(longitude=field["longitude"] % 360).sortby("longitude")
    if field.shape[-2:] != LATLON_SHAPE:
        raise ValueError(f"{name}: grid {field.shape[-2:]} is not the 0.25 degree ERA5 grid {LATLON_SHAPE}")
    return field.values.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="ERA5 (0.25 degree) -> CAST3 C192 initial condition")
    parser.add_argument("inputs", nargs="*", help="ERA5 NetCDF or GRIB files")
    parser.add_argument("--download", action="store_true",
                        help="read ERA5 from ARCO-ERA5 on Google Cloud instead of local files")
    parser.add_argument("--time", required=True, help="initial time, e.g. 2025-04-01T00")
    parser.add_argument("--out", required=True, help="output NetCDF")
    parser.add_argument("--weights", default="remap/era5_to_c192_conserve.npz",
                        help="regridding weights ERA5 -> C192")
    parser.add_argument("--static", default="static.npz", help="C192 grid coordinates")
    return parser.parse_args()


def main():
    args = parse_args()
    init_time = pd.Timestamp(args.time)
    if args.download:
        datasets = [open_arco_era5()]
    elif args.inputs:
        datasets = [open_era5(path) for path in args.inputs]
    else:
        raise SystemExit("give ERA5 files or --download")
    to_c192 = Regridder(args.weights, LATLON_SHAPE, C192_SHAPE)

    data_vars = {}
    for name in PRESSURE_VARS + SURFACE_VARS:
        field = to_c192(read_field(datasets, name, init_time))
        dims = (("level",) if name in PRESSURE_VARS else ()) + C192_DIMS
        data_vars[name] = (dims, field, {"units": UNITS[name]})
        print(f"  {name:26s} {field.min():12.4g} {field.max():12.4g}  {UNITS[name]}")

    grid = np.load(args.static)
    cos_sza = cos_solar_zenith_angle(init_time.to_pydatetime(), grid["lat"], grid["lon"])
    data_vars["cosine_solar_zenith_angle"] = (C192_DIMS, np.asarray(cos_sza, dtype=np.float32))

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": np.datetime64(init_time, "ns"),
            "level": PRESSURE_LEVELS,
            "lat": (C192_DIMS, grid["lat"]),
            "lon": (C192_DIMS, grid["lon"]),
        },
        attrs={"source": "ERA5, conservatively regridded to C192"},
    )
    ds.to_netcdf(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
