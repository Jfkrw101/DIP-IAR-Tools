import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from DIP.intensityTransform import *

class Fourier2D:

    def __init__(self, input_img):
        self.__input_img = input_img
    
    def fft(self):
        # -> Fast Fourier Transform
        fft_complex = fftpack.fft2(self.__input_img)
        # -> Split Magnitude Phase, consequently
        self.__fft_magnitude = np.abs(fft_complex)
        self.__fft_phase = np.arctan2(fft_complex.imag,fft_complex.real)
        # -> Shift Quadrant
        self.__fft_magnitude = fftpack.fftshift(self.__fft_magnitude)


    def ifft(self):
        # - Invert Shift Magnitude
        ifft_magnitude = fftpack.ifftshift(self.__fft_magnitude)
        # - Combine Magnitude Phase
        ifft_real = ifft_magnitude * np.cos(self.__fft_phase)
        ifft_imag = ifft_magnitude * np.sin(self.__fft_phase)
        # - Combine into Complex
        ifft_complex = ifft_real + (ifft_imag * 1j)
        # -> Invert FFT
        output_complex = fftpack.ifft2(ifft_complex)
        # -> Get Image Data from Real part
        self.__output_img = output_complex.real

    def getOutputImg(self):
        return self.__output_img
    

    def showMagnitude(self, ban_radius=1):
        ### - Banning Circle
        center_ban = np.ones_like(self.__fft_magnitude)
        center_ban = np.ascontiguousarray(center_ban)
        # - Center Position
        center = (int(center_ban.shape[1]//2),
        int(center_ban.shape[0]//2))
        # - Draw Circle
        center_ban = cv2.circle(center_ban, center,ban_radius, 0, -1)
        # -> Center Banning
        v_magnitude = self.__fft_magnitude * center_ban
        # -> Log Intensity Transform
        v_magnitude = v_magnitude / v_magnitude.max()
        v_magnitude = logTransform(v_magnitude, c=1,to_uint8=False)
        # ~> Display Magnitude
        plt.imshow(v_magnitude, cmap="hot")
        plt.show()
    
    def getMagnitude(self):
        return self.__fft_magnitude
    
    def setMagnitude(self, fft_magnitude):
        self.__fft_magnitude = fft_magnitude
    
    