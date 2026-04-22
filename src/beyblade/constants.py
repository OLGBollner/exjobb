from scipy import constants as Cn
import numpy as np


CONSTANTS = {
    "ang_amu2SI": np.sqrt(Cn.physical_constants["atomic mass constant"][0]) * 1e-10,
    "meV2rads": 1e-3 * Cn.e / Cn.hbar,
    "MHz2meV": Cn.h * 1e6 / (Cn.e * 1e-3),
    "GHz2meV": Cn.h * 1e9 / (Cn.e * 1e-3),
    "THz2meV": Cn.h * 1e12 / (Cn.e * 1e-3),
    "meV2J": 1e-3 * Cn.e
}
