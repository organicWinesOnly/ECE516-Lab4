import numpy as np
import matplotlib.pyplot as plt
from adap_thresholding import beat_classifier, beat_finder
from polynomial_chirp_trans import PolynomialCT

signal_data = np.loadtxt('../data/ecg_data15.csv', skiprows=105520,
                      max_rows=3980)
sample_rate = 268
signal_data = signal_data - np.mean(signal_data)
pct = PolynomialCT(signal_data, sample_rate, 75, 65, 4)
pct.Run(0.0005, 3)
tfd = np.abs(pct.tfd)
f3 = tfd[3, :]
f4 = tfd[4, :]

# normalize the frequency bands
f3 = f3 / np.max(f3)
f4 = f4 / np.max(f4)

# the sampling rate is modified after the TFA the correct sampling rate is 
# 268 / (75 - 60)
sr_prime = 27

loc3 = beat_finder(f3, sr_prime)
loc4 = beat_finder(f4, sr_prime)
qrs_beats = beat_classifier(f4, f3, sr_prime)
qrs_beats = qrs_beats * 10 #+ 38
ax = plt.subplot(211)
bx = plt.subplot(212)
ax.plot(f3, 'k', label="Third Freq Band")
ax.plot(f4, 'b-', label="Fourth Freq Band")
ax.set_title("Results of beat_finder, for the third and 4th bands")
ax.set_ylabel("Normalized Energy")
ax.set_xlabel("Samples")

ax.plot(np.arange(f3.size)[loc3], f3[loc3], 'ro')
ax.plot(np.arange(f4.size)[loc4], f4[loc4], 'gP')
bx.plot(np.arange(signal_data.size)[qrs_beats], signal_data[qrs_beats], 'kP')
bx.plot(signal_data, 'orange')
bx.set_title("Filtered ECG with labeled QRS complexes")
bx.set_xlabel("Samples")
plt.tight_layout()
plt.show()

