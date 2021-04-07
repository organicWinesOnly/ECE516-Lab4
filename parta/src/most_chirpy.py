import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from adap_thresholding import beat_classifier, beat_finder
from polynomial_chirp_trans import PolynomialCT
from scipy.signal import get_window
from skimage.transform import radon

signal_data = np.loadtxt('../data/ecg_data15.csv', skiprows=1000)
sample_rate = 268
signal_data = signal_data - np.mean(signal_data)
size_of_signal = signal_data.size
num_segments = 50
size_segs = size_of_signal // num_segments
print("Size of segments: {}".format(size_segs))
chirp_rate_values = np.zeros(num_segments, dtype='float')

for i in range(num_segments):
    t = i * size_segs
    # pct becomes a normal chirplet transform for order_poly = 2
    pct = PolynomialCT(signal_data[t: t + size_segs], sample_rate, 75, 65, 2)
    pct.Run(0.0005, 3)
    if pct.alpha.size > 1:
        chirp_rate = pct.alpha[1]
  
        print("Chirp rate between: {} and {}".format(t, t + size_segs))
        print(">>>>> {}".format(chirp_rate))
        chirp_rate_values[i] = chirp_rate

chirpiest = np.argmax(chirp_rate_values) * size_segs
print("The chirpiest segment is between: {} and {}".format(chirpiest,
                                                           chirpiest + size_segs))
                                                           
pcchirpiest = PolynomialCT(signal_data[chirpiest: chirpiest + size_segs],
                          sample_rate, 75, 65, 2)
pct.Run(0.0005, 3)
t = np.arange(0,pct.tfd.shape[1]) * (10 / sample_rate)
f = np.arange(0,pct.tfd.shape[0]) * (sample_rate / 75)
plt.pcolormesh(t,f,np.abs(pct.tfd), cmap=cm.get_cmap('jet'),shading='gouraud')
plt.title("TFD of the Chirpiest portion of the ECG signal")
plt.ylabel("Frequency Hz")
plt.xlabel("Time (s)")
plt.show()
