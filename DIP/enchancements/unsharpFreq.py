import numpy as np
from DIP.fourier import Fourier2D
from DIP.filters.frequency import lowpassFilter


def unsharpFreq(input_img, freq_cutoff, filter_func, n_order=2, k=2):
    ### -> Fast Fourier Transform 2D
    # - Forward FFT
    FFT = Fourier2D(input_img)
    FFT.fft()
    fft_magnitude = FFT.getMagnitude()
    ### -> Laplacian Sharpening
    # -> Create Laplacian Filter
    center_pos = (fft_magnitude.shape[0]//2, fft_magnitude.shape[1]//2)
    lp_filter = lowpassFilter(fft_magnitude.shape[:2], center_pos, freq_cutoff, filter_func, n_order)
    # -> Sharpening
    ifft_magnitude = (1 + k * (1 - lpc_filter)) * fft_magnitude
    # - Inverse FFT
    FFT.setMagnitude(ifft_magnitude)
    FFT.ifft()
    output_img = FFT.getOutputImg()
    ### -> Clip range into [0, 255]
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    return output_img