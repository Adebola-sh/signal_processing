
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
duration = 10  # seconds
sampling_rate = 1000  # Hz
frequency = 2  # Hz (frequency of the ultrasonic signal)
amplitude = 10  # Amplitude of the ultrasonic signal
noise_amplitude = 3  # Amplitude of noise

csv_file_path = 'mixed_ultrasonic_data.csv'

# Check if the CSV file exists
if os.path.exists(csv_file_path):
    # Read data from CSV
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        t, mixed_signal = zip(*[(float(row[0]), float(row[1])) for row in csv_reader])
else:
    # Time array
    t = np.linspace(0, duration, sampling_rate * duration, endpoint=False)

    # Generate ultrasonic signal (sine wave)
    ultrasonic_signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # Add random noise
    noise = noise_amplitude * np.random.normal(size=len(t))

    # Mix the signals
    mixed_signal = ultrasonic_signal + noise

    # Save data to CSV
    csv_data = list(zip(t, mixed_signal))

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Time', 'Mixed Signal'])
        csv_writer.writerows(csv_data)

# Plot the graph
plt.figure(figsize=(12, 6))
plt.plot(t, mixed_signal, label='Mixed Signal', alpha=0.7)
plt.title('Mixed Ultrasonic Signal with Noise')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
