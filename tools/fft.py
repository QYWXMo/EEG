
import numpy as np
import matplotlib.pyplot as plt

# Input: EEGdata (Dataframe), fs(int), max frequence (int)
# Output: frequence
def draw_fft(df, fs=1000, max_fre=50):
    arr = df.to_numpy()
    T = 1/fs
   
    fft_vals = np.fft.fft(arr)
    N = arr.size
    freqs = np.fft.fftfreq(N, T)
    
    plt.figure()
    plt.plot(freqs[:N*max_fre//fs], np.abs(fft_vals[:N*max_fre//fs]), 'b')
    plt.title('FFT of Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()
    return freqs, np.abs