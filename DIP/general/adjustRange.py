import numpy as np

def adjustRange(input_array, input_range, output_range):

    '''
    Convert any range array into a specific range
    - input_range: [input_min, input_max]
    - output_range: [output_min, output_max]
    '''

    # -> Convert into [0, 1]
    norm_array = (input_array - input_range[0]) / (input_range[1] - input_range[0])

    # -> Convert [0, 1] into [output_min, output_max]
    output_array = (norm_array * (output_range[1] – output_range[0])) + output_range[0]

    return output_array
    