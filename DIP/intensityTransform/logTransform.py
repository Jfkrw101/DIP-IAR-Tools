import numpy as np

def logTransform(input_array, c=None, to_uint8=True):
    # -> Default C
    if c == None:
        max_level = np.iinfo(input_array.dtype).max - np.iinfo(input_array.dtype).min
        c = max_level / (np.log(1 + np.max(input_array)))
    # -> Convert "input_array", prevent overflow
    input_array = input_array.astype(float)
    # -> Log Transform
    trans_array = c * np.log(1 + input_array)
    if to_uint8:
        output_array = trans_array.astype(np.uint8)
    else:
        output_array = trans_array
        
    return output_array