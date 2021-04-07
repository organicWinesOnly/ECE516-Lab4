""" adap_thresholding.py

    Based on:
        Jiapu Pan and Willis J. Tompkins, A Real-Time QRS Detection Algorithm,
        IEEE Transactions on Biomedical Engineering, BME-32(3), pp. 230–236,
        March (1985).

        B. S. Shaik, G. V. S. S. K. R. Naganjaneyulu, T. Chandrasheker, and A.
        Narasimhadhan, “A method for QRS delineation based on stft using
        adaptive threshold,” Procedia Computer Science, vol. 54, pp. 646–653,
        2015.
"""
import numpy as np 
from scipy.signal import get_window, stft, find_peaks, find_peaks_cwt
import matplotlib.pyplot as plt

class RRQueue:
    def __init__(self, cond=False):
        self.Q = []
        self.cond = cond

    def size(self):
        return len(self.Q)

    def rravg(self):
        if self.size() < 2:
            return 0
        else:
            sum_ = np.zeros(len(self.Q) -1)
            for i in range(len(self.Q) - 1):
                sum_[i] = self.Q[i+1] - self.Q[i]
            return np.mean(sum_)

    def pop(self):
        """ remove top of queue.
        """
        self.Q.pop(0)

    def push(self, item):
        if not self.cond:
            if len(self.Q) == 8:
                self.pop()
            self.Q.append(item)

        elif len(self.Q) >=2:
            item_prime = item - self.Q[-1]
            rr_avg2 = self.rravg()
            if item_prime > 0.92 * rr_avg2 and item_prime < 1.16 * rr_avg2:
                if len(self.Q) == 8:
                    self.pop()
                self.Q.append(item)
        else:
            self.Q.append(item)


def threshold(peak, spki, npki):
    spk = 0.125 * peak + 0.875 * spki
    npk = 0.125 * peak + 0.875 * npki
    T1 = npk + 0.25 * (spk - npk)
    T2 = 0.9 * T1
    return T1, T1

def next_loc(peaks, loc, fs):
    n_loc = loc + 200 * (fs / 1000)
    return np.searchsorted(peaks, int(n_loc))
    
    
def beat_finder(freq_comp, fs):
    """ Returns the locations of the QRS complexes in freq_comp

        ==== Params ====
        freq_compx: 1D np.ndarry of time dependent energy values
        fs: int, sampling rate in sex
    """
    
    signal_threshold = 0.05
    noise_threshold = 10e-3
    peak = np.max(freq_comp)
    rr1 = RRQueue()  # holds the latest 8 beats
    rr2 = RRQueue(True)  # holds latest 8 beats w/ value of 92-122% of the RR interval avg
    beat_locations = []  # the returned value
    beat_cache = RRQueue()  # peaks that are not above t1 but above t2

    # calculate the first and second threshold
    T1, T2 = threshold(peak, signal_threshold, noise_threshold)
    peak_loc = find_peaks(freq_comp, signal_threshold)[0]
    last_loc = peak_loc[0]  # last peak location
    loc = peak_loc[0]  # current peak location

    i = 0
    while i < peak_loc.size:
        loc = peak_loc[i]
        if freq_comp[loc] >= T1:
            beat_locations.append(loc)
            rr1.push(loc)
            rr2.push(loc)
            # update thresholds iff the estimated signal peak ahas changed
            if signal_threshold <= freq_comp[loc]:
                signal_threshold = freq_comp[loc]
                T1, T2 = threshold(peak, signal_threshold, noise_threshold)
            last_loc = loc
            # refractory condition
            i = next_loc(peak_loc, loc, fs)

        # Back check if outside the rr2 missed limit: 1.66 * rr2.rravg
        elif last_loc - loc > 1.66 * rr2.rravg():
            # check if the current location should be added to the cache
            if freq_comp[loc] >= T2:
                beat_cache.push(loc)

            # check if there are suitable candidates in the cache
            if beat_cache.size() > 0:
                top_beat = beat_cache.pop()
            else:
                top_beat = -1
            while beat_cache.size > 0:
                beat = beat_cache.pop()
                if freq_comp[beat] >= freq_comp[top_beat]:
                    top_beat = beat

            # iff there were beats in the cache, add the largest beat to the
            # list of locations and update the thresholds
            if top_beat != -1:
                beat_locations.append(top_beat)
                rr1.push(top_beat)
                rr2.push(top_beat)
                if signal_threshold <= freq_comp[top_beat]:
                    signal_threshold = freq_comp[top_beat]
                    T1, T2 = threshold(peak, signal_threshold, noise_threshold)
                last_loc = loc
                # refractory condition
                i = next_loc(peak_loc, loc, fs)
            else:
                i+=1

        else:
            if freq_comp[loc] >= T2:
                beat_cache.push(loc)
            # this is a peak due to noise
            else:
                if noise_threshold <= freq_comp[loc]:
                    noise_threshold = freq_comp[loc]
                    T1, T2 = threshold(peak, signal_threshold, noise_threshold)
            i += 1
    return np.array(beat_locations)

def beat_classifier(f1, f2, fs):
    loc1 = beat_finder(f1, fs)
    loc2 = beat_finder(f2, fs)
    matching =  []
    
    for element in loc1:
        peak_range = np.arange(element -5, element+5)
        check = np.isin(peak_range, loc2)
        if np.any(check):
            x = int(peak_range[check])
            mid = np.round((x + element) / 2)
            matching.append(int(mid))
    return np.array(matching)



if __name__ == '__main__':
    # np.random.seed(123)
    # def s(t):
    #     return np.sin(2*np.pi * t)

    # time = np.linspace(0, 10, 800)
    # fs = 100
    # ss = s(time) + np.random.normal(0, 0.3, time.size)
    # loc = beat_finder(ss, fs)
    # plt.plot(time, ss, 'k')
    # plt.plot(time[loc], ss[loc], 'D')
    # plt.show()
    # print(loc)
    sig = np.loadtxt("data/butt_filt_data.csv", delimiter=",", skiprows=1000,
                        max_rows=20000)
    sig = sig / np.max(sig)
    loc = beat_finder(sig, 268)
    # loc = find_peaks(sig)[0]
    plt.plot(sig, 'k')
    plt.plot(np.arange(sig.size)[loc], sig[loc], 'rD')
    plt.show()
