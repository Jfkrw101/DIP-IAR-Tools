import cv2

def gaussianFilter(filter_size):
    
    '''
    Gaussian Filter
    '''

    # -> Create Gaussian Filter
    gauss_filter = cv2.getGaussianKernel(filter_size, -1)
    gauss_filter = gauss_filter * gauss_filter.T

    return gauss_filter