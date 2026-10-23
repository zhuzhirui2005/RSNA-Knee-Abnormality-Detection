import numpy as np
def native_volume(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)
