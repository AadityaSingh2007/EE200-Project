import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, welch, spectrogram, freqz, tf2zpk
import requests
from io import BytesIO
from IPython.display import Audio


def fetch_audio_from_drive(url):
    """Fetch the audio file content from Google Drive link."""
    response = requests.get(url)
    return BytesIO(response.content)


def preprocess_audio(buffer):
    """Read and normalize WAV audio from a buffer."""
    rate, data = wavfile.read(buffer)
    data = data.astype(float)
    data /= np.max(np.abs(data))
    return rate, data


def summarize_audio(rate, data):
    """Display sampling details and duration."""
    length_sec = len(data) / rate
    print(f"Sample Rate: {rate} Hz")
    print(f"Audio Length: {length_sec:.2f} seconds")


def visualize_psd(data, rate, label="Power Spectrum"):
    """Show power spectral density using Welch method."""
    freqs, psd_values = welch(data, rate, nperseg=4096)
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd_values)
    plt.title(label)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.show()
    return freqs, psd_values


def build_band_reject_filters(rate, bands):
    """Create multiple notch filters for given frequency bands."""
    filters = []
    nyquist = 0.5 * rate

    for low, high in bands:
        norm_low = low / nyquist
        norm_high = high / nyquist
        b, a = butter(N=4, Wn=[norm_low, norm_high], btype='bandstop')
        filters.append((b, a))

    return filters


def show_filter_response(filters, rate):
    """Plot gain response of all designed filters."""
    nyquist = 0.5 * rate
    for idx, (b, a) in enumerate(filters, 1):
        freq, h = freqz(b, a, worN=2000)
        plt.figure(figsize=(10, 5))
        plt.plot(freq * nyquist / np.pi, 20 * np.log10(abs(h)))
        plt.title(f'Filter {idx} Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.grid(True)
        plt.show()


def draw_pole_zero_map(filters):
    """Plot pole-zero diagrams."""
    for idx, (b, a) in enumerate(filters, 1):
        zeros, poles, _ = tf2zpk(b, a)
        plt.figure(figsize=(6, 6))
        plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zeros')
        plt.scatter(np.real(poles), np.imag(poles), marker='x', label='Poles')
        plt.title(f'Pole-Zero Plot {idx}')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.grid(True)
        plt.legend()
        plt.show()


def clean_audio_with_filters(data, filters):
    """Apply all band-reject filters sequentially."""
    output = data.copy()
    for b, a in filters:
        output = filtfilt(b, a, output)
    return output


def compare_before_after_psd(raw, clean, rate):
    """Compare PSDs before and after filtering."""
    f1, psd1 = welch(raw, rate, nperseg=4096)
    f2, psd2 = welch(clean, rate, nperseg=4096)
    plt.figure(figsize=(10, 5))
    plt.semilogy(f1, psd1, label='Original')
    plt.semilogy(f2, psd2, label='Filtered')
    plt.title('PSD Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.legend()
    plt.show()


def compare_spectrograms(raw, clean, rate):
    """Plot spectrograms before and after filtering."""
    for title, data in [('Original', raw), ('Filtered', clean)]:
        freq, t, Sxx = spectrogram(data, rate)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, freq, 10 * np.log10(Sxx))
        plt.title(f'Spectrogram ({title})')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.tight_layout()
        plt.show()


def export_clean_audio(data, rate, filename='filtered_output.wav'):
    """Save filtered audio to disk in 16-bit PCM format."""
    if np.max(np.abs(data)) < 1e-4:
        print("⚠️ Filtered signal is very quiet.")

    norm = np.int16(data / np.max(np.abs(data)) * 32767)
    wavfile.write(filename, rate, norm)
    print(f"✅ Clean audio written to '{filename}'")
    return norm


def run_pipeline():
    """Run the full audio demixing pipeline."""
    # Step 1: Download from Google Drive
    link = 'https://drive.google.com/uc?export=download&id=1J0LF5Nz3tFwGXSgI0HjmCqa3pJi2ks1u'
    buffer = fetch_audio_from_drive(link)

    # Step 2: Load and normalize
    rate, raw_data = preprocess_audio(buffer)
    summarize_audio(rate, raw_data)

    # Step 3: Listen to input (only in Jupyter)
    print("▶ Playing original audio...")
    print(Audio(raw_data, rate=rate))

    # Step 4: PSD analysis
    visualize_psd(raw_data, rate)

    # Step 5: Define and create filters
    unwanted_bands = [
        (1000, 2200),
        (2300, 3700),
        (3000, 5600),
        (5200, 8100)
    ]
    filters = build_band_reject_filters(rate, unwanted_bands)

    # Step 6 & 7: Visual diagnostics
    show_filter_response(filters, rate)
    draw_pole_zero_map(filters)

    # Step 8: Apply filters
    cleaned = clean_audio_with_filters(raw_data, filters)

    # Step 9–10: Compare output
    compare_before_after_psd(raw_data, cleaned, rate)
    compare_spectrograms(raw_data, cleaned, rate)

    # Step 11: Save result
    export_clean_audio(cleaned, rate)

    # Step 12: Listen to cleaned result (Jupyter only)
    print("▶ Cleaned audio ready:")
    print(Audio(cleaned, rate=rate))


# Run the updated pipeline
if __name__ == "__main__":
    run_pipeline()
