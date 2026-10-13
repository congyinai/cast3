# CAST3 C192 — ONNX inference

Global 10-day weather forecast with the CAST3 diffusion model on the C192 cubed-sphere grid
(6 faces × 192 × 192, 12-hourly steps), packaged as an ONNX model so that it runs without the
model source code.

## Files

| File | What it is |
|---|---|
| `infer.py` | Forecast script: onnxruntime for the network, `diffusers` for the DDIM scheduler, `earthkit-meteo` for the solar zenith angle |
| `static.npz` | Grid lat/lon, static input fields (topography, sin(lat), cell area), normalisation mean/std |
| `requirements.txt` | Python dependencies |

The model itself is **not** part of this repository. Put these two files next to `infer.py`:

| File | What it is |
|---|---|
| `model.onnx` | Network graph, float32 (2.5 MB) |
| `model.onnx.data` | Network weights (2.5 GB), loaded automatically by `model.onnx` |

## Install

```bash
pip install -r requirements.txt
```

A CUDA GPU with about 20 GB of memory is needed. `onnxruntime-gpu` must match the CUDA version:
releases up to 1.26 use CUDA 12, newer ones CUDA 13.

## Run

```bash
python infer.py --init 2025040100.nc --out forecast.nc
```

| Option | Default | Meaning |
|---|---|---|
| `--init` | (required) | initial-condition NetCDF |
| `--out` | `forecast.nc` | output NetCDF |
| `--days` | 10 | forecast length |
| `--ddim-steps` | 20 | DDIM denoising steps per 12 h step |
| `--seed` | 0 | diffusion noise seed; different seeds give different ensemble members |
| `--device` | 0 | GPU id |

A 10-day forecast takes about 6 minutes on one RTX 5090.

## Input

One NetCDF file on the C192 grid with dimensions `(nf=6, Ydim=192, Xdim=192)`, a scalar `time`, and

- on pressure levels (`level` must contain 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa):
  `geopotential`, `specific_humidity`, `temperature`, `u_component_of_wind`, `v_component_of_wind`
- at the surface: `mean_sea_level_pressure`, `10m_u_component_of_wind`, `10m_v_component_of_wind`,
  `2m_temperature`, `cosine_solar_zenith_angle`

## Output

The same nine forecast variables in physical units on the same grid, with dimensions
`(time, level, nf, Ydim, Xdim)` or `(time, nf, Ydim, Xdim)`; `time` holds the valid times
(+12 h, +24 h, …) and `lat` / `lon` give the cell centres.
