import numpy as np
from midi_processing import RANGES

RANGES_SUM = np.cumsum(RANGES)

class MusicAugmentations:
    def __init__(self, transpose=(-3,3), time_stretch=(0.95,0.975,1.0,1.025,1.05)):
        """
        Class for applying random transpose and time_stretch augmentations for encoded sequences.
        
        Parameters
        ----------
        transpose : tuple(min, max)
            range for transpose in pitches.
        time_stretch : list
            list of time_stretch multipliers to sample from.
        """
        self.transpose = range(transpose[0], transpose[1]+1)
        self.time_stretch = time_stretch
    
    def __call__(self, encoded):
        """encoded: list or 1D np.ndarray"""
        transpose = np.random.choice(self.transpose)
        # time_stretch = np.random.uniform(*self.time_stretch)
        time_stretch = np.random.choice(self.time_stretch)
        return augment(encoded, transpose, time_stretch)
        
def augment(encoded, transpose, time_stretch):
    """
    Applies transpose and time_stretch augmentation for encoded sequence. Inplace operation.

    Parameters
    ----------
    encoded : np.ndarray or list
        encoded sequence (input for model).
    transpose : int
        bias for transpose in pitches.
    time_stretch : float
        time_stretch multiplier.
        
    Returns
    -------
    encoded : np.array or list
        augmented sequence.
    """
    for i,ev in enumerate(encoded):
        if ev < RANGES_SUM[0]:
            # NOTE_ON
            encoded[i] = min(RANGES_SUM[0]-1, max(0, ev+transpose))
        elif ev < RANGES_SUM[1]:
            # NOTE_OFF
            encoded[i] = min(RANGES_SUM[1]-1, max(RANGES_SUM[0], ev+transpose))
        elif ev < RANGES_SUM[2]:
            # SET_VELOCITY
            pass
        elif ev < RANGES_SUM[3] and time_stretch != 1.0:
            # TIME_SHIFT
            t = ev - RANGES_SUM[2] + 1  # since 0 = 10ms
            t = max(min(RANGES[3], int(round(t*time_stretch))), 1)
            encoded[i] = t + RANGES_SUM[2] - 1
        else:
            continue
    return encoded