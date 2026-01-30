"""
Sorting algorithms for adversarial audio dataset partitioning.

These functions rank audio samples by various criteria (loudness, spectral
features, rhythm, timbre) to enable train-test splits that maximize dissimilarity.

All functions accept either file paths or pre-loaded audio arrays.
Requires librosa for audio processing.
"""

import io
from pathlib import Path

import librosa
import numpy as np


def _load_audio(audio, sr=22050):
    """
    Load audio from path or return array directly.

    Args:
        audio: file path, or tuple of (samples, sample_rate)
        sr: target sample rate if loading from file

    Returns:
        Tuple of (samples, sample_rate)
    """
    if isinstance(audio, (str, Path)):
        y, sr = librosa.load(audio, sr=sr)
        return y, sr
    elif isinstance(audio, tuple):
        return audio
    else:
        # Assume it's already a numpy array, use default sr
        return audio, sr


# =============================================================================
# Loudness / Energy
# =============================================================================

def mean_amplitude(audios, sr=22050, low_first=True):
    """
    Sort audio by mean absolute amplitude.

    Useful for adversarial splits: train on mid-volume audio,
    test on very quiet or very loud audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, quieter audio first; if False, louder first

    Returns:
        List of (index, amplitude) tuples sorted by mean amplitude.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, _ = _load_audio(audio, sr)
        amp = np.abs(y).mean()
        scores.append((i, amp))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def rms_energy(audios, sr=22050, low_first=True):
    """
    Sort audio by root mean square energy.

    RMS energy is a common measure of audio loudness that better
    correlates with perceived volume than simple amplitude.

    Useful for adversarial splits: train on consistent energy levels,
    test on very quiet or very loud audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, lower energy first; if False, higher energy first

    Returns:
        List of (index, rms) tuples sorted by RMS energy.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, _ = _load_audio(audio, sr)
        rms = np.sqrt(np.mean(y ** 2))
        scores.append((i, rms))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def dynamic_range(audios, sr=22050, low_first=True):
    """
    Sort audio by dynamic range (difference between max and min amplitude).

    Higher dynamic range indicates more variation between loud and quiet parts.
    Lower dynamic range indicates more compressed/consistent audio.

    Useful for adversarial splits: train on compressed audio,
    test on highly dynamic audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, compressed audio first; if False, dynamic audio first

    Returns:
        List of (index, range) tuples sorted by dynamic range.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, _ = _load_audio(audio, sr)
        dyn_range = np.abs(y).max() - np.abs(y).min()
        scores.append((i, dyn_range))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def peak_to_average_ratio(audios, sr=22050, low_first=True):
    """
    Sort audio by peak-to-average ratio (crest factor).

    Higher values indicate peaky/transient audio (e.g., percussion).
    Lower values indicate more sustained/compressed audio.

    Useful for adversarial splits: train on sustained sounds,
    test on transient/percussive sounds.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, sustained audio first; if False, peaky audio first

    Returns:
        List of (index, ratio) tuples sorted by peak-to-average ratio.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, _ = _load_audio(audio, sr)
        peak = np.abs(y).max()
        avg = np.abs(y).mean()
        ratio = peak / avg if avg > 0 else 0
        scores.append((i, ratio))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


# =============================================================================
# Frequency / Spectral
# =============================================================================

def spectral_centroid(audios, sr=22050, low_first=True):
    """
    Sort audio by spectral centroid ("center of mass" of the spectrum).

    Higher values indicate brighter, treble-heavy audio.
    Lower values indicate darker, bass-heavy audio.

    Useful for adversarial splits: train on mid-frequency audio,
    test on very bright or very dark audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, darker audio first; if False, brighter audio first

    Returns:
        List of (index, centroid_hz) tuples sorted by spectral centroid.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
        mean_centroid = centroid.mean()
        scores.append((i, mean_centroid))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def spectral_bandwidth(audios, sr=22050, low_first=True):
    """
    Sort audio by spectral bandwidth (spread around the centroid).

    Higher values indicate wider frequency spread (broadband).
    Lower values indicate narrower frequency content (tonal).

    Useful for adversarial splits: train on narrow-band audio,
    test on broadband audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, narrow-band first; if False, broadband first

    Returns:
        List of (index, bandwidth_hz) tuples sorted by spectral bandwidth.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)
        mean_bandwidth = bandwidth.mean()
        scores.append((i, mean_bandwidth))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def spectral_rolloff(audios, sr=22050, roll_percent=0.85, low_first=True):
    """
    Sort audio by spectral rolloff frequency.

    The rolloff frequency is the frequency below which a specified
    percentage of total spectral energy lies.

    Higher values indicate more high-frequency content.
    Lower values indicate more low-frequency content.

    Useful for adversarial splits: train on one frequency profile,
    test on different frequency profiles.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        roll_percent: energy percentage threshold (default 0.85)
        low_first: if True, low rolloff first; if False, high rolloff first

    Returns:
        List of (index, rolloff_hz) tuples sorted by spectral rolloff.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sample_rate, roll_percent=roll_percent
        )
        mean_rolloff = rolloff.mean()
        scores.append((i, mean_rolloff))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def spectral_flatness(audios, sr=22050, low_first=True):
    """
    Sort audio by spectral flatness (Wiener entropy).

    Values close to 1 indicate noise-like audio (flat spectrum).
    Values close to 0 indicate tonal audio (peaked spectrum).

    Useful for adversarial splits: train on tonal audio,
    test on noisy audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, tonal audio first; if False, noisy audio first

    Returns:
        List of (index, flatness) tuples sorted by spectral flatness.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        mean_flatness = flatness.mean()
        scores.append((i, mean_flatness))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def zero_crossing_rate(audios, sr=22050, low_first=True):
    """
    Sort audio by zero crossing rate.

    Higher values indicate noisier or more percussive audio.
    Lower values indicate smoother, more tonal audio.

    Useful for adversarial splits: train on smooth audio,
    test on noisy/percussive audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, smooth audio first; if False, noisy audio first

    Returns:
        List of (index, zcr) tuples sorted by zero crossing rate.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, _ = _load_audio(audio, sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = zcr.mean()
        scores.append((i, mean_zcr))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def fundamental_frequency(audios, sr=22050, low_first=True):
    """
    Sort audio by estimated fundamental frequency (pitch).

    Uses pYIN algorithm for robust pitch estimation.

    Useful for adversarial splits: train on mid-pitch audio,
    test on very high or very low pitch audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, lower pitch first; if False, higher pitch first

    Returns:
        List of (index, f0_hz) tuples sorted by fundamental frequency.
        Audio with no detectable pitch receives f0 of 0.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate
        )
        # Take mean of voiced frames only
        voiced_f0 = f0[~np.isnan(f0)]
        mean_f0 = voiced_f0.mean() if len(voiced_f0) > 0 else 0
        scores.append((i, mean_f0))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


# =============================================================================
# Timbre / MFCCs
# =============================================================================

def mfcc_mean(audios, sr=22050, n_mfcc=13, low_first=True):
    """
    Sort audio by mean of first MFCC coefficient.

    The first MFCC coefficient relates to overall energy/loudness.
    Higher MFCCs capture timbral characteristics.

    Useful for adversarial splits: separate by timbral characteristics.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        n_mfcc: number of MFCC coefficients (default 13)
        low_first: if True, lower mean MFCC first

    Returns:
        List of (index, mfcc_mean) tuples sorted by mean of first MFCC.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
        # Use mean of first coefficient (energy-related)
        mean_mfcc = mfccs[0].mean()
        scores.append((i, mean_mfcc))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def mfcc_variance(audios, sr=22050, n_mfcc=13, low_first=True):
    """
    Sort audio by variance of MFCC coefficients over time.

    Higher variance indicates more timbral variation over time.
    Lower variance indicates more consistent timbre.

    Useful for adversarial splits: train on stable timbre,
    test on varying timbre.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        n_mfcc: number of MFCC coefficients (default 13)
        low_first: if True, stable timbre first; if False, varying timbre first

    Returns:
        List of (index, mfcc_var) tuples sorted by MFCC variance.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
        # Compute mean variance across all coefficients
        var = mfccs.var(axis=1).mean()
        scores.append((i, var))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


# =============================================================================
# Rhythm / Music
# =============================================================================

def tempo(audios, sr=22050, low_first=True):
    """
    Sort audio by estimated tempo (beats per minute).

    Useful for adversarial splits: train on mid-tempo audio,
    test on very slow or very fast audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, slower tempo first; if False, faster tempo first

    Returns:
        List of (index, bpm) tuples sorted by tempo.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        bpm, _ = librosa.beat.beat_track(y=y, sr=sample_rate)
        scores.append((i, float(bpm)))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def beat_strength(audios, sr=22050, low_first=True):
    """
    Sort audio by beat/onset strength.

    Higher values indicate more percussive, rhythmically strong audio.
    Lower values indicate more ambient, less rhythmic audio.

    Useful for adversarial splits: train on strong beat audio,
    test on ambient/weak beat audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, weak beats first; if False, strong beats first

    Returns:
        List of (index, strength) tuples sorted by beat strength.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)
        mean_strength = onset_env.mean()
        scores.append((i, mean_strength))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def harmonic_ratio(audios, sr=22050, low_first=True):
    """
    Sort audio by harmonic-to-percussive ratio.

    Higher values indicate more melodic/harmonic content.
    Lower values indicate more percussive/rhythmic content.

    Useful for adversarial splits: train on melodic audio,
    test on percussive audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, percussive audio first; if False, harmonic audio first

    Returns:
        List of (index, ratio) tuples sorted by harmonic ratio.
    """
    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)
        harmonic, percussive = librosa.effects.hpss(y)
        h_energy = np.sum(harmonic ** 2)
        p_energy = np.sum(percussive ** 2)
        ratio = h_energy / (p_energy + 1e-10)  # Avoid division by zero
        scores.append((i, ratio))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


# =============================================================================
# Quality
# =============================================================================

def compression_ratio(audios, sr=22050, low_first=True):
    """
    Sort audio by compression ratio (uncompressed / compressed size).

    Simple/repetitive audio compresses well (high ratio).
    Complex/varied audio compresses poorly (low ratio).

    Useful for adversarial splits: train on simple audio,
    test on complex audio.

    Args:
        audios: list of file paths or (samples, sr) tuples
        sr: sample rate for loading files (default 22050)
        low_first: if True, complex audio first; if False, simple audio first

    Returns:
        List of (index, ratio) tuples sorted by compression ratio.
    """
    import soundfile as sf

    scores = []
    for i, audio in enumerate(audios):
        y, sample_rate = _load_audio(audio, sr)

        # Uncompressed size (samples * bytes per sample)
        uncompressed_size = len(y) * 4  # float32 = 4 bytes

        # Compress to OGG in memory
        buffer = io.BytesIO()
        sf.write(buffer, y, sample_rate, format='OGG')
        compressed_size = buffer.tell()

        ratio = uncompressed_size / compressed_size if compressed_size > 0 else 0
        scores.append((i, ratio))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores
