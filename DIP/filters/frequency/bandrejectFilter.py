from DIP.filters.frequency import bandpassFilter

def bandrejectFilter(filter_size, filter_pos, band_center, band_width, filter_func, n_order=2):
    # -> Create Band-reject filter from Band-pass filter
    br_filter = 1 - bandpassFilter(filter_size, filter_pos, band_center, band_width, filter_func,n_order)
    
    return br_filter