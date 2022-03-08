import numpy as np


def lowpass(data, n = 101):
    """Averages over n data points.
    """
    
    return(np.array([np.mean(data[i : i + n]) for i in range(data.size - n + 1)]))


def highpass(data, n = 101):
    """Filter out low-frequency drift.
    Offsets each datapoint by the average of the surrounding n data points. 
    N must be odd.
    """
    
    if(n % 2 != 1):
        raise Exception("n must be odd!")
    return(np.array(data[int((n - 1) / 2) : -int((n - 1) / 2)] - lowpass(data, n)))
