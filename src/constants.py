import numpy as np

# * Parameters
# Physical
ALPHA = 1
HEAT_SOURCE_INTENSITY = np.pi ** 2

# Real physical parameters (of aluminum)
THERMAL_CONDUCTIVITY = 0.23                                         # W / (mm K)
DENSITY = 2700 * 1e-9                                               # kg / mm^3
SPECIFIC_HEAT_CAPACITY = 900                                        # J / (kg K)
THERMAL_DIFFUSIVITY = THERMAL_CONDUCTIVITY \
                      / (DENSITY * SPECIFIC_HEAT_CAPACITY) * 1e-3   # in ms

# Space
REFERENCE_LENGTH = 1
NUM_OF_SPATIAL_STEPS = 40
dx = REFERENCE_LENGTH / (NUM_OF_SPATIAL_STEPS - 1)
X_DOMAIN = np.linspace(0, REFERENCE_LENGTH, NUM_OF_SPATIAL_STEPS)

# Time
REFERENCE_TIME = 1
dt = 0.49 * dx ** 2 / ALPHA # inverse stability
NUM_OF_TIME_STEPS = int(REFERENCE_TIME / dt)
TIME_STEPS = np.arange(0, NUM_OF_TIME_STEPS)
TIME_DOMAIN = np.linspace(0, REFERENCE_TIME, NUM_OF_TIME_STEPS)

# Stability condition, (deprecated)
if ALPHA * dt / dx**2  > 0.5:
    print(ALPHA * dt / dx ** 2 )
    raise ValueError("Stability condition not met.")
