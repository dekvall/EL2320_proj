# Various parameters for the simulation, JPDA and Particle filter
import numpy as np

# Number of simulated particles
N_PARTICLES = 800

# Probability of detection
P_D = 0.95

# Probability of a false positive
P_FA = 0.1

# Measurement noise
R = 0.2**2 * np.eye(2)

# Process noise
P = 0.3**2 * np.eye(4)

# X velocity Standard deviation of introduced "bounce" noise
BOUNCE_XV_STDDEV = 0.2

# Y velocity Standard deviation of introduced "bounce" noise
BOUNCE_YV_STDDEV = 0.2
