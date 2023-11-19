import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# Load your data from Excel into a pandas DataFrame
# Assuming time and amplitude columns in columns 0 and 1
data = pd.read_excel('mixed_ultra_sonic_sensor_data_excel.xlsx')

# Identify sampling frequency
time_interval = data.iloc[1, 0] - data.iloc[0, 0]
original_sampling_frequency = 1 / time_interval

# Set a target sampling frequency (adjust as needed)
target_sampling_frequency = 2 * 100

# Calculate the downsample factor
downsample_factor = int(original_sampling_frequency / target_sampling_frequency)

# Downsample the data
downsampled_data = data.iloc[::downsample_factor, :]

# Quantization parameters
num_bits = 8  # Number of bits for quantization

# Perform quantization
quantization_levels = 2**num_bits
max_amplitude = np.max(downsampled_data.iloc[:, 1])
min_amplitude = np.min(downsampled_data.iloc[:, 1])
amplitude_range = max_amplitude - min_amplitude

# Map amplitudes to quantization levels
quantized_values = np.round(((downsampled_data.iloc[:, 1] - min_amplitude) / amplitude_range) * (quantization_levels - 1))

# Map quantized values back to the original amplitude range
quantized_amplitudes = (quantized_values / (quantization_levels - 1)) * amplitude_range + min_amplitude

# Perform FFT on the quantized signal
fft_result = np.fft.fft(quantized_amplitudes)
fft_freq = np.fft.fftfreq(len(quantized_amplitudes), d=time_interval)

# Design an FIR low-pass filter
cutoff_frequency = 0.1  # Adjust as needed
num_taps = 51  # Adjust as needed, should be an odd number
fir_filter = firwin(num_taps, cutoff_frequency / (0.5 * original_sampling_frequency), window='hamming')

# Apply the FIR filter using convolution
filtered_signal = lfilter(fir_filter, 1.0, quantized_amplitudes)

# Plot the original, downsampled, quantized signals, FFT, and filtered signal
plt.figure(figsize=(14, 10))

plt.subplot(5, 1, 1)
plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original Signal')
plt.title('Original Signal')

plt.subplot(5, 1, 2)
plt.stem(downsampled_data.iloc[:, 0], downsampled_data.iloc[:, 1], label='Downsampled Signal')
plt.title('Downsampled Signal')

plt.subplot(5, 1, 3)
plt.step(downsampled_data.iloc[:, 0], quantized_amplitudes, label='Quantized Signal')
plt.title('Quantized Signal')

plt.subplot(5, 1, 4)
plt.stem(fft_freq, np.abs(fft_result))
plt.title('FFT of Quantized Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.subplot(5, 1, 5)
plt.step(downsampled_data.iloc[:, 0], filtered_signal, label='Filtered Signal', color='green')
plt.title('Filtered Signal')

plt.tight_layout()
plt.show()
