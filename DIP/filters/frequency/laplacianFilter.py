import numpy as np
from DIP.general import adjustRange


def laplacianFunction(distance_map, freq_cutoff):
    lpc_func = 4*(np.pi)**2 * distance_map**2
    return lpc_func

def distanceMap(size, pos):
    map_height = size[0]
    map_width = size[1]
    # -> Preset Position Index
    v = np.arange(0.5*((map_height+1)%2),0.5*((map_height+1)%2)+map_height)
    u = np.arange(0.5*((map_width+1)%2),0.5*((map_width+1)%2)+map_width)
    uv, vv = np.meshgrid(u, v)
    # -> Distance Map
    distance_map = ((uv-pos[1])**2 + (vv-pos[0])**2)**0.5
    return distance_map

def laplacianFilter(filter_size, filter_pos):
    # -> Distance Map 2D from given position 'filter_pos'
    distance_map = distanceMap(filter_size, filter_pos)
    # -> Create Frequency Filter from Laplacian Function
    lpc_filter = laplacianFunction(distance_map)
    # -> Convert [0, ~) to [0, 1]
    lpc_filter = adjustRange(lpc_filter,(lpc_filter.min(), lpc_filter.max()),(0, 1))

    return lpc_filter