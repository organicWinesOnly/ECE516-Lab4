# Study of the "Chirpyness" of an ECG signal

In order to calculate the chirpyness of the ECG signal I implemented the
Polynomial Chirplet Transform (PCT) algorithm described in [1] \(see
polynomial_chirp_trans.py\). The only difference is that the threshold check has
no been completed -- hence the threshold value for the 'Run' method plays no
part in the output of the system. The program is designed to work for any order
polynomial but in order to calculate the chirpyness the system was built and ran
on a polynomial of order 2. A gaussian window of size 75 was used with an
overlap of 65 units. These specific sizes were used due to the effectiveness of
previous work in using PCT for ECG signal analysis [2].

### Preprocessing
The ECG data from the file 'ECOG_15' was passed through a low pass butterworth
filter with cutoff of 40 Hz, passband ripple of 3 dB, and stop band attenuation
of 60 dB. The goal of the filter was to remove any artifacts due to power line
interference [3]. See filter_ecg_data.py for the python implementation.

### Analysis
The first 1000 data points were ignored due to its messy behaviour. The
remaining data sectioned into 50 segments of 3980 samples. For each segment the
PCT was performed for 3 iterations using the parameters described above. Section
of data that was found to be the most chirpy were samples in the range 
(95520, 99500) which corresponds to the time 1:00:01 to 1:02:30.

Chirpiest TFD:
![](chirpiest_tfd.png)

Other chirping segments included:
1. (0:03:05, 0:05:34)
2. (0:05:34, 0:08:02)
3. (0:10:31, 0:13:00)
4. (0:55:04, 0:57:33)
5. (0:57:33, 1:00:01)
6. (1:00:01, 1:02:30)
7. (1:02:30, 1:04:59)
8. (1:04:59, 1:07:27)
9. (1:09:55, 1:12:24)
10. (1:12:24, 1:14:53)

### ff plane computations
Using ff.m >> a matlab file created by Steve Mann to build a Frequency Frequency
plot, the FF planes for all the above segments.
![](ff-slices/seg1.jpg)
![](ff-slices/seg2.jpg)
![](ff-slices/seg3.jpg)
![](ff-slices/seg4.jpg)
![](ff-slices/seg5.jpg)
![](ff-slices/seg6.jpg)
![](ff-slices/seg7.jpg)
![](ff-slices/seg8.jpg)
![](ff-slices/seg9.jpg)
![](ff-slices/seg10.jpg)

### Heartbeat detector (Bonus)
Using the time-frequency distribution (TFD) calculated from a (PCT) it became
possible to complete the work suggested in [2] to develop a QRS complex
detector for a given ECG waveform. An adaptive threshold was used to find peaks
in the 3 and 4 frequency bands of and ECG TFD. The TFD was created using the
same window and overlap described above. The 3rd and 4th frequency bands were
chosen because these contained the frequencies relevant to an ECG signal [2].
The adaptive threshold was initially suggested in [4] and slightly altered in
[2]. A peak in an ECG signal is considered a beak if and only if the QRS
complexes found in the 3rd and 4th energy match (to some degree of error). For
the implementation of the threshold see 'adap_threshold.py'. For an example of
how to find QRS complexes see, 'sample_heartbeat_classifier.py'. Some results
for the chirpiest waveform are shown below.

![](QRS_complex_results.png)

### References
[1] Z. Peng, G. Meng, F. Chu, Z. Lang, W. Zhang, and Y. Yang, “Poly- nomial
chirplet transform with application to instantaneous frequency estimation,” IEEE
Transactions on Instrumentation and Measurement, vol. 60, no. 9, pp. 3222–3229,
2011.

[2] G. V. S. S. K. R. Naganjaneyulu, B. S. Shaik and A. V. Narasimhadhan, "R
peak delineation in ECG signal based on polynomial chirplet transform using
adaptive threshold," 2016 11th International Conference on Industrial and
Information Systems (ICIIS), Roorkee, India, 2016, pp. 856-860, doi:
10.1109/ICIINFS.2016.8263058.

[3] B. S. Shaik, G. V. S. S. K. R. Naganjaneyulu and A. V. Narasimhadhan, "A
novel approach for QRS delineation in ECG signal based on chirplet transform,"
2015 IEEE International Conference on Electronics, Computing and Communication
Technologies (CONECCT), Bangalore, India, 2015, pp. 1-5, doi:
10.1109/CONECCT.2015.7383914.

[4] J. Pan and W. J. Tompkins, “A real-time QRS detection algorithm,” IEEE
Transactions on Biomedical Engineering, vol. BME-32, pp. 230– 236, March 1985.
