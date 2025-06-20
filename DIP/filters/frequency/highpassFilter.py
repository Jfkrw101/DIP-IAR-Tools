import numpy as np
from DIP.filters.frequency import lowpassFilter

def highpassFilter(filter_size, filter_pos, freq_cutoff, filter_func, n_order=2):
    # -> Create High-pass filter from Low-pass filter
    hp_filter = 1 - lowpassFilter(filter_size, filter_pos, freq_cutoff, filter_func,n_order)
    
    return lp_filter