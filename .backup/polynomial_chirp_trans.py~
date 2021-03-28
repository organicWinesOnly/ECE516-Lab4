""" Algorithm to calculate the Polynomial Chirp Transformation parameters as 
    described by:
	Polynomial Chirplet Transform With Application to Instantaneous
	Frequency Estimation Z. K. Peng, G. Meng, F. L. Chu, Z. Q. Lang, W. M.
	Zhang, and Y. Yang
"""

import numpy as np
import matplotlib.pyplot as plt
from math import inf
from scipy.integrate import romb
from scipy.signal import convolve, stft
from scipy.signal.windows import gaussian
from numpy.polynomial.polynomial import Polynomial
import matplotlib.cm as cm

#  complete
def GaussianWindow(times, sigma):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (times / sigma) ** 2)

def TesterFunc(data, alpha):
    sum_ = np.zeros(data.size)
    for i in range(alpha.size):
        sum_ = sum_ + alpha[i] * data ** i
    return sum_

class PolynomialCT:
    """  Calculate the Polynomial Chirp Transformation parameters
    """
    # not
    def __init__(self, data, initial_params=None):
        """ Initialize class. 
        """ 
        if initial_params is None:
            alpha = np.random.randint(-5, 5, 21)
        else:
            alpha = alpha
        self.alpha = alpha
        self.z = data
        self._polynomial = Polynomial(alpha)  
        # current frequency values for PCT pct:(time,freq)-> (time, freq)
        self._freq = np.linspace(-1,1, 2+2 ** 6, endpoint=False)[1:]
        # transform frequency
        self._freq = self._freq / (1 - self._freq ** 2)
        self._current_pct = np.zeros((self.z.size, self._freq.size),
                                     dtype='complex')
        self._freq_integral_factor =(1 + self._freq ** 2)/ ((1 - self._freq ** 2) ** 2)

    # complete
    def _FRotationOperator(self, times):
        """ Calculate the nonlinear Frequency Rotation operator for a given dataset
            <z>, using parameters <alpha>.
        """
        sum_ = np.zeros(times.shape, dtype='complex')
        for i in range(2, self._polynomial.degree() + 2):
            sum_ = sum_ + 1/ i * self.alpha[i-2] * times ** i
        return np.exp(-1j * sum_)

    # complete
    def _FShiftOperator(self, times, t0):
        """ Calculate the nonlinear Frequency Shift operator for a given dataset <z>,
            using parameters <alpha>.
        """
        sum_ = np.zeros(times.shape, dtype='complex')
        for i in range(2, self._polynomial.degree() + 2):
            sum_ = sum_ + self.alpha[i-2] * t0 ** (i-1) * times
        return np.exp(1j * sum_)
    
    # complete
    def REN(self, pct):
        # frequency correction due to infinitite integral
        integrand = np.log10(np.power(np.abs(pct), 3))
        return -1 * romb(romb(integrand, axis=1), axis=0)

    def PCT(self, initial_state=False):
        """ Calculate the polynomial chirp transformation using the parameters 
            <alpha> for  a analytic signal z[n] <z>.

            This function updates the following attributes: polynomial, ,
            _current_pct, and alpha
            === params ===

            alpha: array of paramaters for the transform
            z: analytic signal data
        """
        # update polynomial
        print("PCT running...")
        if not initial_state: 
            # recalculate the params of the instantaneous velocity
            # generate a line
            y_fit = np.max(np.abs(self._current_pct), axis=0)
            x_fit = np.arange(y_fit.size)
            fitted_polynomial = self._polynomial.fit(x_fit,
                                                     y_fit,
                                                     self._polynomial.degree())
            self._polynomial = fitted_polynomial.convert()
            self.alpha = self._polynomial.coef
            plt.plot(x_fit, y_fit, 'r.')
            plt.plot(x_fit, TesterFunc(x_fit/ 268 , self.alpha), 'g')
            plt.show()

        # transform signal and compute the STFT
        times = np.arange(self.z.size) / 268  # divide by sampling rate
        freq_rot = self._FRotationOperator(times)
        window = ('gaussian', np.std(self.z))  # gaussian window
        freq_shift = self._FShiftOperator(times)
        f_t = self.z * freq_rot * freq_shift
        # complete the stft with gaussian window
        _, _, self._current_pct = stft(f_t, fs=268, window=window,nperseg=33)
        print(self._current_pct.shape)
        print(times.shape)
        

        #freq_time = np.outer(self._freq, times)
        # for t0 in range(times.size):
        #     freq_rot = self._FRotationOperator(times)
        #     freq_shift = self._FShiftOperator(times, times[t0])
        #     window = gaussian(2048, np.std(self.z.size)/ 2) # gaussian window
        #     window = window.reshape((-1,1))
        #     # integrate
        #     print(window.ndim)
        #     f_t = self.z * freq_rot * freq_shift * np.exp(-1j * freq_time)
        #     print(f_t.ndim)
        #     # convolve transformed signal with window
        #     self._current_pct[t0, :] = convolve(f_t, window)
        print("PCT end")

    # complete
    def Run(self, err, max_iter=1000):
        """ Run the system.

            === params ===
            max_iter: maximum number of runs
            err: stopping condition form 0 < err < 1
        """
        change_in_ren = inf
        count = 1
        self.PCT(True)
        while (change_in_ren > err) and (max_iter > count):
            print(count)
            prev_pct = self._current_pct.copy()
            self.PCT()
            curr_pct = self._current_pct
            REN_prev = self.REN(prev_pct)
            REN_curr = self.REN(curr_pct)
            change_in_ren = np.abs((REN_curr - REN_prev) / REN_curr) 
            count += 1
if __name__ == "__main__":
    sig_data = np.loadtxt('data/butt_filt_data.csv', skiprows=1000,
                          max_rows=4340)
    pctTest = PolynomialCT(sig_data)
    pctTest.Run(0.0005, 5)
