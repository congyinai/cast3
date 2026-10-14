"""Variables, grids and regridding shared by prepare_init.py and infer.py."""
import numpy as np
import scipy.sparse

# ======================== Variables ========================
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
UNITS = {
    "geopotential": "m2 s-2",
    "specific_humidity": "kg kg-1",
    "temperature": "K",
    "u_component_of_wind": "m s-1",
    "v_component_of_wind": "m s-1",
    "mean_sea_level_pressure": "Pa",
    "10m_u_component_of_wind": "m s-1",
    "10m_v_component_of_wind": "m s-1",
    "2m_temperature": "K",
}
N_LEVELS = len(PRESSURE_LEVELS)                               # 13
N_PRESSURE_CHANNELS = len(PRESSURE_VARS) * N_LEVELS           # 65
N_STATE_CHANNELS = N_PRESSURE_CHANNELS + len(SURFACE_VARS)    # 69

# ======================== Grids ========================
# C192 cubed sphere: 6 faces x 192 x 192 cells
C192_DIMS = ("nf", "Ydim", "Xdim")
C192_SHAPE = (6, 192, 192)

# ERA5 regular latitude-longitude grid, 0.25 degree
LATLON_DIMS = ("latitude", "longitude")
LATLON_SHAPE = (721, 1440)
LATITUDES = np.linspace(90, -90, 721)       # north to south, as in ERA5
LONGITUDES = np.arange(1440) * 0.25         # 0 ... 359.75


class Regridder:
    """First-order conservative regridding with precomputed weights (remap/*.npz).

    The weights were generated with the ESMF offline regridding weight generator and converted with
    remap/make_npz.py. They form a sparse matrix W (entries S at 0-based indices row, col), and
    regridding is the matrix product  target = W @ source  on the flattened grids.
    """

    def __init__(self, weight_file, source_shape, target_shape):
        w = np.load(weight_file)
        weights, rows, cols = w["S"], w["row"], w["col"]
        shape = (int(np.prod(target_shape)), int(np.prod(source_shape)))
        self.matrix = scipy.sparse.csr_matrix((weights, (rows, cols)), shape=shape, dtype=np.float32)
        self.source_shape = tuple(source_shape)
        self.target_shape = tuple(target_shape)

    def __call__(self, fields):
        """fields: (..., *source_shape) -> (..., *target_shape), float32."""
        leading = fields.shape[:fields.ndim - len(self.source_shape)]
        flat = fields.reshape(-1, self.matrix.shape[1]).astype(np.float32)
        out = (self.matrix @ flat.T).T
        return np.ascontiguousarray(out).reshape(leading + self.target_shape)
