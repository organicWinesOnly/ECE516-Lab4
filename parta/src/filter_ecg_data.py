""" Filter the ECG data.
"""
from scipy.signal import *
import numpy as np
import matplotlib.pyplot as plt
from math import inf

#start = 215650  + 12000 # usable data starts here
data = np.loadtxt("../data/ecg_data15.csv", delimiter=",")
                  #skiprows=start, max_rows=12000)

# =============================================================================
# Build butterworth filer (bandpass)
# second order butterworth lowpass filter with 40 Hz cutoff
# =============================================================================
fs = 268  # sampling rate in Hz
nyq_limit = 0.5 * fs
lowpass_cutoff = 40 /nyq_limit  # Hz
passband_ripple = 3.0 # dB
stopband_attenuation = 60.0 # dB

# Calculate the correct order of the filter
order, wn = buttord(lowpass_cutoff, 0.05 + lowpass_cutoff,
                    gpass=passband_ripple,
                    gstop=stopband_attenuation)

# convert bounds to units of fs
# upper_limit = upper_bound / nyq_limit
# lower_limit = lower_bound / nyq_limit
 
sos = butter(order, wn,
              btype='lowpass',
              output='sos')

filt = sosfilt(sos, data)
# plt.plot(data, 'g')
plt.plot(filt, 'r')
plt.title("butterworth filter")
plt.show()
np.savetxt("../data/butt_filt_data.csv", filt, delimiter=",")
