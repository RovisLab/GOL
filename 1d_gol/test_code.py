import numpy as np
import matplotlib.pyplot as plt

data = np.ones((300))

max_idx = 100
indices = np.arange(0, np.shape(data)[0], 1)

print(indices)

scaling_offset = np.absolute(indices - max_idx)

scaling_offset = scaling_offset / np.max(scaling_offset)

print(scaling_offset)