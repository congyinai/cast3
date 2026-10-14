<p align="center">
  <img src="assets/logo.png" alt="CAST3" width="240">
</p>

# CAST3 C192 — ONNX inference

Global 10-day weather forecast with the CAST3 diffusion model on the C192 cubed-sphere grid
(6 faces × 192 × 192, 12-hourly steps), packaged as an ONNX model so that it runs without the
model source code. Initial conditions are prepared from ERA5, and forecasts are written on the
0.25° ERA5 latitude-longitude grid (or on the native C192 grid).

## Files

| File | What it is |
|---|---|
| `prepare_init.py` | ERA5 (0.25°) → C192 initial condition |
| `infer.py` | Forecast: onnxruntime for the network, `diffusers` for the DDIM scheduler, `earthkit-meteo` for the solar zenith angle |
| `common.py` | Variable list, grid definitions and the regridding used by both scripts |
| `remap/` | Conservative regridding weights ERA5 → C192 (`era5_to_c192_conserve.npz`) and C192 → ERA5 grid (`c192_to_era5_conserve.npz`), and `make_npz.py` that made them |
| `static.npz` | C192 grid lat/lon, static input fields (topography, sin(lat), cell area), normalisation mean/std |
| `requirements.txt` | Python dependencies |

## Download the model and the test data

The model parameters and a sample initial condition are **not** part of this repository. They are on Google Drive:

| What | Google Drive folder | Files |
|---|---|---|
| Model parameters | [parameter](https://drive.google.com/drive/folders/1olUrGP9VdJE_xIyLSJc1dbkdDU9QZAWd) | `model.onnx` (network graph, 2.5 MB), `model.onnx.data` (float32 weights, 2.5 GB) |
| Test data | [data](https://drive.google.com/drive/folders/1zVZjoLrfLt_qJgqxTtXmMGQ8fH5OEiQj) | `2025040100.nc` (C192 initial condition, 2025-04-01 00 UTC, 67 MB) |

Both folders are in <https://drive.google.com/drive/folders/1SjOpyRy-aqo_wKyN9_-o3xMn_wOOUqCS?usp=sharing>.

Put the files of `parameter` next to `infer.py` (do not rename them; `model.onnx` loads the weights
by file name) and the test data in `data/`. From the command line:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1olUrGP9VdJE_xIyLSJc1dbkdDU9QZAWd -O .       # model parameters
gdown --folder https://drive.google.com/drive/folders/1zVZjoLrfLt_qJgqxTtXmMGQ8fH5OEiQj -O data    # test data
```

The regridding weights in `remap/` are first-order conservative weights generated with the ESMF offline
regridding weight generator, stored compactly (only the sparse weight matrix, float32) by `remap/make_npz.py`.

## Install

```bash
pip install -r requirements.txt
```

A CUDA GPU with about 20 GB of memory is needed. `onnxruntime-gpu` must match the CUDA version:
releases up to 1.26 use CUDA 12, newer ones CUDA 13.

## Run

**1. Initial condition.** The test file `data/2025040100.nc` is ready to use. For another date,
download ERA5 for that time from Google Cloud and convert it to C192:

```bash
python prepare_init.py --time 2025-04-01T00 --out data/2025040100.nc --download
```

`--download` reads [ARCO-ERA5](https://github.com/google-research/arco-era5), the public copy of ERA5
(0.25°, hourly) on Google Cloud Storage. No account is needed; about 1 GB is downloaded per initial
time. Dates up to about one week before today are available (the last weeks are ERA5T, the
preliminary ERA5).

ERA5 files you already have (NetCDF or GRIB, 0.25°, any number of files) can be used instead:

```bash
python prepare_init.py --time 2025-04-01T00 --out data/2025040100.nc  era5_pressure.nc era5_surface.nc
```

The files must together contain, at that time,

- on pressure levels 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa:
  geopotential, specific humidity, temperature, u and v wind
- at the surface: mean sea level pressure, 10 m u and v wind, 2 m temperature

Other variables, levels and times in the files are ignored.

**2. Forecast.**

```bash
python infer.py --init data/2025040100.nc --out forecast.nc
```

| Option | Default | Meaning |
|---|---|---|
| `--init` | (required) | C192 initial-condition NetCDF |
| `--out` | `forecast.nc` | output NetCDF |
| `--grid` | `latlon` | output grid: `latlon` (0.25° ERA5 grid) or `c192` (native cubed sphere) |
| `--days` | 10 | forecast length |
| `--ddim-steps` | 20 | DDIM denoising steps per 12 h step |
| `--seed` | 0 | diffusion noise seed; different seeds give different ensemble members |
| `--device` | 0 | GPU id |

A 10-day forecast takes about 6 minutes on one RTX 5090.

## Output

Nine variables in physical units: geopotential, specific_humidity, temperature,
u_component_of_wind, v_component_of_wind on the 13 pressure levels, and mean_sea_level_pressure,
10m_u_component_of_wind, 10m_v_component_of_wind, 2m_temperature. `time` holds the valid times
(+12 h, +24 h, …).

- `--grid latlon`: dimensions `(time, level, latitude, longitude)` on the 721 × 1440 ERA5 grid
  (latitude 90 → −90, longitude 0 → 359.75), conservatively regridded from C192.
  A 10-day forecast is about 3.4 GB (7 days: 2.3 GB).
- `--grid c192`: dimensions `(time, level, nf, Ydim, Xdim)`; `lat` / `lon` give the cell centres.
