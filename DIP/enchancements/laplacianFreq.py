import numpy as np
from DIP.fourier import Fourier2D
from DIP.filters.frequency import laplacianFilter

def laplacianFreq(input_img, k=2):
    ### -> Fast Fourier Transform 2D
    # - Forward FFT
    FFT = Fourier2D(input_img)
    FFT.fft()
    fft_magnitude = FFT.getMagnitude()
    ### -> Laplacian Sharpening
    # -> Create Laplacian Filter
    center_pos = (fft_magnitude.shape[0]//2, fft_magnitude.shape[1]//2)
    lpc_filter = laplacianFilter(fft_magnitude.shape[:2], center_pos)
    # -> Sharpening
    ifft_magnitude = fft_magnitude + k * (fft_magnitude * lpc_filter)
    # - Inverse FFT
    FFT.setMagnitude(ifft_magnitude)
    output_img = FFT.getOutputImg()
    FFT.ifft()
    ### -> Clip range into [0, 255]
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img
