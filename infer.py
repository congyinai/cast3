"""CAST3 C192 global weather forecast (ONNX Runtime inference).

Usage
-----
    python infer.py --init 2025040100.nc --out forecast.nc

Method
------
CAST3 is a conditional diffusion model on the C192 cubed-sphere grid (6 faces x 192 x 192 cells).
The forecast advances in 12 h steps. For every step:

  1. Build the conditioning: current atmospheric state + solar zenith angle + static fields.
  2. Draw Gaussian noise and denoise it with DDIM. The result is the 12 h *increment* of the state.
  3. Add the increment to the current state. The new state conditions the next step.

Channels
--------
State, 69 channels, always in this order:

     0-12  geopotential            on 13 pressure levels
    13-25  specific_humidity       on 13 pressure levels
    26-38  temperature             on 13 pressure levels
    39-51  u_component_of_wind     on 13 pressure levels
    52-64  v_component_of_wind     on 13 pressure levels
    65     mean_sea_level_pressure
    66     10m_u_component_of_wind
    67     10m_v_component_of_wind
    68     2m_temperature

Conditioning, 73 channels: the 69 state channels, then the cosine of the solar zenith angle,
surface geopotential (topography), sin(latitude) and grid-cell area.

Units and orientation
---------------------
The network works with normalised fields ((x - mean) / std per channel) and with every cube face
flipped along Ydim relative to the NetCDF files. Inside this script, arrays named `*_norm` are in
that network convention; everything read from or written to disk is in physical units and file
orientation. `to_network()` and `to_physical()` convert between the two.
"""
import argparse
import datetime as dt
import time

import numpy as np
import onnxruntime as ort
import torch
import xarray as xr
from diffusers import DDIMScheduler
from earthkit.meteo.solar import cos_solar_zenith_angle

# ======================== Model constants ========================
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]   # hPa
PRESSURE_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
SURFACE_VARS = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]
GRID_DIMS = ("nf", "Ydim", "Xdim")
GRID_SHAPE = (6, 192, 192)
N_LEVELS = len(PRESSURE_LEVELS)                         # 13
N_PRESSURE_CHANNELS = len(PRESSURE_VARS) * N_LEVELS     # 65
N_STATE_CHANNELS = N_PRESSURE_CHANNELS + len(SURFACE_VARS)   # 69
STEP_HOURS = 12


# ======================== Units and orientation ========================

def flip_ydim(field):
    """Flip the Ydim axis (second to last). Converts file orientation <-> network orientation."""
    return field[..., ::-1, :]


def to_network(state, static):
    """Physical state (69, 6, 192, 192) in file orientation -> normalised, network orientation."""
    mean = static["mean"][:, None, None, None]
    std = static["std"][:, None, None, None]
    return flip_ydim((state - mean) / std).astype(np.float32)


def to_physical(state_norm, static):
    """Inverse of `to_network`."""
    mean = static["mean"][:, None, None, None]
    std = static["std"][:, None, None, None]
    return (flip_ydim(state_norm) * std + mean).astype(np.float32)


def solar_channel(valid_time, static):
    """Cosine of the solar zenith angle at `valid_time`, network orientation, shape (1, 6, 192, 192)."""
    cos_sza = cos_solar_zenith_angle(valid_time, static["lat"], static["lon"])
    return flip_ydim(np.asarray(cos_sza, dtype=np.float32))[None]


# ======================== Input / output ========================

def load_static(path):
    """Read static.npz.

    lat, lon : (6, 192, 192)     cell-centre coordinates in degrees
    static   : (3, 6, 192, 192)  standardised topography, sin(latitude), standardised cell area
    mean, std: (69,)             normalisation constants of the state channels
    """
    return dict(np.load(path))


def read_initial_condition(path):
    """Read the initial state from a C192 NetCDF file.

    Returns
    -------
    state     : (69, 6, 192, 192) float32, physical units, file orientation
    cos_sza   : (1, 6, 192, 192)  cosine of the solar zenith angle stored in the file
    init_time : datetime.datetime
    """
    with xr.open_dataset(path) as ds:
        ds = ds.sel(level=PRESSURE_LEVELS)

        channels = []
        for name in PRESSURE_VARS:
            channels.append(ds[name].values.reshape(N_LEVELS, *GRID_SHAPE))
        for name in SURFACE_VARS:
            channels.append(ds[name].values.reshape(1, *GRID_SHAPE))
        state = np.concatenate(channels, axis=0)

        cos_sza = ds["cosine_solar_zenith_angle"].values.reshape(1, *GRID_SHAPE)
        init_time = ds["time"].values.reshape(-1)[0].astype("datetime64[s]").astype(dt.datetime)

    return state.astype(np.float32), cos_sza.astype(np.float32), init_time


def write_forecast(path, states, valid_times, init_time, static):
    """Write the forecast to NetCDF.

    states: (n_times, 69, 6, 192, 192), physical units, file orientation.
    """
    data_vars = {}
    for i, name in enumerate(PRESSURE_VARS):
        first = i * N_LEVELS
        data_vars[name] = (("time", "level") + GRID_DIMS, states[:, first:first + N_LEVELS])
    for i, name in enumerate(SURFACE_VARS):
        data_vars[name] = (("time",) + GRID_DIMS, states[:, N_PRESSURE_CHANNELS + i])

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": np.array(valid_times, dtype="datetime64[ns]"),
            "level": PRESSURE_LEVELS,
            "lat": (GRID_DIMS, static["lat"]),
            "lon": (GRID_DIMS, static["lon"]),
        },
        attrs={"model": "CAST3 C192", "init_time": init_time.isoformat()},
    )
    compression = {name: {"zlib": True, "complevel": 1} for name in data_vars}
    ds.to_netcdf(path, encoding=compression)


# ======================== Diffusion model ========================

def load_network(model_path, gpu_id):
    """Open model.onnx with ONNX Runtime on the given GPU.

    The network maps (x, t, cond) -> v:
        x    (1, 69, 6, 192, 192)  noisy 12 h increment
        t    (1,)                  diffusion timestep, 0..999
        cond (1, 73, 6, 192, 192)  conditioning
        v    (1, 69, 6, 192, 192)  v-prediction
    """
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()      # find the CUDA / cuDNN libraries installed through pip
    providers = [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    network = ort.InferenceSession(model_path, providers=providers)
    if network.get_providers()[0] != "CUDAExecutionProvider":
        print("WARNING: CUDA is not available to onnxruntime, running on CPU (very slow)")
    return network


def build_scheduler(ddim_steps):
    """DDIM scheduler with the noise schedule used in training."""
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        clip_sample=False,
        prediction_type="v_prediction",
    )
    scheduler.set_timesteps(ddim_steps)
    return scheduler


def sample_increment(network, scheduler, cond, generator):
    """Sample the normalised 12 h increment with DDIM (eta = 0).

    cond: (1, 73, 6, 192, 192) float32. Returns (69, 6, 192, 192) float32.
    """
    sample = torch.randn((1, N_STATE_CHANNELS) + GRID_SHAPE, generator=generator)

    for t in scheduler.timesteps:
        inputs = {
            "x": scheduler.scale_model_input(sample, t).numpy(),
            "t": np.array([float(t)], dtype=np.float32),
            "cond": cond,
        }
        v = network.run(None, inputs)[0]
        sample = scheduler.step(torch.from_numpy(v), t, sample, eta=0).prev_sample

    return sample[0].numpy()


# ======================== Forecast ========================

def run_forecast(network, scheduler, static, state, cos_sza, init_time, n_steps, generator):
    """Autoregressive rollout.

    state, cos_sza: initial condition in physical units and file orientation.
    Returns the forecast states (n_steps, 69, 6, 192, 192) in physical units and their valid times.
    """
    state_norm = to_network(state, static)                         # (69, 6, 192, 192)
    solar_norm = flip_ydim(cos_sza)                                # ( 1, 6, 192, 192)
    static_norm = flip_ydim(static["static"]).astype(np.float32)   # ( 3, 6, 192, 192)

    states, valid_times = [], []
    for step in range(1, n_steps + 1):
        start = time.time()
        valid_time = init_time + dt.timedelta(hours=step * STEP_HOURS)

        cond = np.concatenate([state_norm, solar_norm, static_norm])[None]    # (1, 73, 6, 192, 192)
        increment_norm = sample_increment(network, scheduler, cond, generator)
        state_norm = state_norm + increment_norm

        # the next step is conditioned on the sun position at the time of the state just predicted
        solar_norm = solar_channel(valid_time, static)

        states.append(to_physical(state_norm, static))
        valid_times.append(valid_time)
        print(f"  +{step * STEP_HOURS:3d} h  {valid_time:%Y-%m-%d %H}Z  {time.time() - start:5.1f} s",
              flush=True)

    return np.stack(states), valid_times


def parse_args():
    parser = argparse.ArgumentParser(description="CAST3 C192 global weather forecast")
    parser.add_argument("--init", required=True, help="C192 initial-condition NetCDF")
    parser.add_argument("--out", default="forecast.nc", help="output NetCDF")
    parser.add_argument("--days", type=float, default=10, help="forecast length in days")
    parser.add_argument("--ddim-steps", type=int, default=20,
                        help="DDIM denoising steps per 12 h step")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the diffusion noise; each seed gives one ensemble member")
    parser.add_argument("--model", default="model.onnx", help="ONNX model")
    parser.add_argument("--static", default="static.npz", help="grid, static fields, normalisation")
    parser.add_argument("--device", type=int, default=0, help="GPU id")
    return parser.parse_args()


def main():
    args = parse_args()
    n_steps = int(round(args.days * 24 / STEP_HOURS))

    static = load_static(args.static)
    state, cos_sza, init_time = read_initial_condition(args.init)
    network = load_network(args.model, args.device)
    scheduler = build_scheduler(args.ddim_steps)
    generator = torch.Generator().manual_seed(args.seed)

    print(f"init {init_time:%Y-%m-%d %H}Z | {n_steps} x {STEP_HOURS} h | "
          f"{args.ddim_steps} DDIM steps | seed {args.seed}")
    states, valid_times = run_forecast(network, scheduler, static, state, cos_sza, init_time,
                                       n_steps, generator)

    write_forecast(args.out, states, valid_times, init_time, static)
    print("saved", args.out)


if __name__ == "__main__":
    main()
