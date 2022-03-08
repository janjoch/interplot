import numpy as np


def lowpass(data, n = 101):
    """Average array over n data points.
    N must be odd.
    """

    if(n % 2 != 1):
        raise Exception("n must be odd!")
    n -= 1
    return(np.array([np.mean(data[i : i + n + 1]) for i in range(data.size - n)]))


def highpass(data, n = 101):
    """Filter out low-frequency drift.
    Offsets each datapoint by the average of the surrounding n data points. 
    N must be odd.
    """
    
    if(n % 2 != 1):
        raise Exception("n must be odd!")
    return(np.array(data[int(n/2) : -int(n/2)] - lowpass(data, n)))
