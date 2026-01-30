"""Generate test audio files for testing audio_sorters.py"""

import numpy as np
import soundfile as sf
from pathlib import Path

output_dir = Path("test_data/audio")
output_dir.mkdir(parents=True, exist_ok=True)

SR = 22050  # Sample rate
DURATION = 1.0  # Duration in seconds


def save_audio(samples, name):
    """Save numpy array as WAV file."""
    # Normalize to prevent clipping
    if np.abs(samples).max() > 0:
        samples = samples / np.abs(samples).max() * 0.9
    sf.write(output_dir / f"{name}.wav", samples.astype(np.float32), SR)
    print(f"Created {name}.wav")


def generate_sine(freq, duration=DURATION, amplitude=1.0):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_noise(duration=DURATION, amplitude=1.0):
    """Generate white noise."""
    samples = int(SR * duration)
    return amplitude * np.random.randn(samples)


# =============================================================================
# Loudness / Energy variations
# =============================================================================

# Quiet sine wave
quiet = generate_sine(440, amplitude=0.1)
save_audio(quiet, "loudness_quiet")

# Medium sine wave
medium = generate_sine(440, amplitude=0.5)
save_audio(medium, "loudness_medium")

# Loud sine wave
loud = generate_sine(440, amplitude=1.0)
save_audio(loud, "loudness_loud")


# =============================================================================
# Dynamic range variations
# =============================================================================

# Low dynamic range (consistent amplitude)
t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
low_dynamic = 0.5 * np.sin(2 * np.pi * 440 * t)
save_audio(low_dynamic, "dynamic_low")

# High dynamic range (amplitude envelope)
envelope = np.abs(np.sin(2 * np.pi * 2 * t))  # Modulate at 2 Hz
high_dynamic = envelope * np.sin(2 * np.pi * 440 * t)
save_audio(high_dynamic, "dynamic_high")


# =============================================================================
# Spectral centroid variations (brightness)
# =============================================================================

# Low frequency (dark/bass) - 100 Hz
dark = generate_sine(100)
save_audio(dark, "spectral_dark")

# High frequency (bright/treble) - 4000 Hz
bright = generate_sine(4000)
save_audio(bright, "spectral_bright")

# Mid frequency - 1000 Hz
mid = generate_sine(1000)
save_audio(mid, "spectral_mid")


# =============================================================================
# Spectral flatness variations (tonal vs noisy)
# =============================================================================

# Pure tone (low flatness)
tonal = generate_sine(440)
save_audio(tonal, "flatness_tonal")

# White noise (high flatness)
noisy = generate_noise()
save_audio(noisy, "flatness_noisy")


# =============================================================================
# Spectral bandwidth variations
# =============================================================================

# Narrow band (single sine)
narrow = generate_sine(1000)
save_audio(narrow, "bandwidth_narrow")

# Broadband (noise)
broad = generate_noise()
save_audio(broad, "bandwidth_broad")


# =============================================================================
# Zero crossing rate variations
# =============================================================================

# Low ZCR (low frequency sine)
low_zcr = generate_sine(50)
save_audio(low_zcr, "zcr_low")

# High ZCR (high frequency sine + noise)
high_zcr = generate_sine(8000) + 0.3 * generate_noise()
save_audio(high_zcr, "zcr_high")


# =============================================================================
# Fundamental frequency (pitch) variations
# =============================================================================

# Low pitch - 100 Hz
low_pitch = generate_sine(100)
save_audio(low_pitch, "pitch_low")

# High pitch - 1000 Hz
high_pitch = generate_sine(1000)
save_audio(high_pitch, "pitch_high")


# =============================================================================
# MFCC variance variations (timbre stability)
# =============================================================================

# Stable timbre (constant sine)
stable_timbre = generate_sine(440)
save_audio(stable_timbre, "timbre_stable")

# Varying timbre (frequency sweep)
t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
freq_sweep = np.linspace(200, 2000, len(t))
varying_timbre = np.sin(2 * np.pi * freq_sweep * t / SR * np.cumsum(np.ones(len(t))))
# Simpler approach: chirp
varying_timbre = np.sin(2 * np.pi * (200 + 1800 * t / DURATION) * t)
save_audio(varying_timbre, "timbre_varying")


# =============================================================================
# Tempo variations
# =============================================================================

# Slow tempo (~60 BPM) - clicks every 1 second
t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)  # 2 seconds
slow_tempo = np.zeros_like(t)
for beat_time in [0.0, 1.0]:
    beat_idx = int(beat_time * SR)
    if beat_idx < len(slow_tempo) - 100:
        slow_tempo[beat_idx:beat_idx+100] = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.005, 100)) * np.exp(-np.linspace(0, 5, 100))
save_audio(slow_tempo, "tempo_slow")

# Fast tempo (~180 BPM) - clicks every 0.33 seconds
t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
fast_tempo = np.zeros_like(t)
for beat_time in np.arange(0, 2.0, 0.33):
    beat_idx = int(beat_time * SR)
    if beat_idx < len(fast_tempo) - 100:
        fast_tempo[beat_idx:beat_idx+100] = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.005, 100)) * np.exp(-np.linspace(0, 5, 100))
save_audio(fast_tempo, "tempo_fast")


# =============================================================================
# Beat strength variations
# =============================================================================

# Strong beats (loud clicks)
t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
strong_beats = np.zeros_like(t)
for beat_time in np.arange(0, 2.0, 0.5):
    beat_idx = int(beat_time * SR)
    if beat_idx < len(strong_beats) - 200:
        # Loud, sharp click
        strong_beats[beat_idx:beat_idx+200] = np.sin(2 * np.pi * 500 * np.linspace(0, 0.01, 200)) * np.exp(-np.linspace(0, 10, 200))
save_audio(strong_beats, "beats_strong")

# Weak/no beats (ambient sine)
weak_beats = generate_sine(440, duration=2.0)
save_audio(weak_beats, "beats_weak")


# =============================================================================
# Harmonic ratio variations
# =============================================================================

# Harmonic (sustained tone with harmonics)
t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
harmonic = (np.sin(2 * np.pi * 440 * t) +
            0.5 * np.sin(2 * np.pi * 880 * t) +
            0.25 * np.sin(2 * np.pi * 1320 * t))
save_audio(harmonic, "harmonic_melodic")

# Percussive (clicks/impulses)
percussive = np.zeros(int(SR * DURATION))
for i in range(0, len(percussive), int(SR * 0.1)):
    if i + 50 < len(percussive):
        percussive[i:i+50] = np.random.randn(50) * np.exp(-np.linspace(0, 5, 50))
save_audio(percussive, "harmonic_percussive")


# =============================================================================
# Compression ratio variations
# =============================================================================

# Simple (single sine - compresses well)
simple = generate_sine(440)
save_audio(simple, "compress_simple")

# Complex (noise - compresses poorly)
complex_audio = generate_noise()
save_audio(complex_audio, "compress_complex")


print(f"\nGenerated {len(list(output_dir.glob('*.wav')))} test audio files in {output_dir}/")
