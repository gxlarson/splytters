"""Unit tests for audio_sorters.py"""

import pytest
from pathlib import Path

from sorters.audio_sorters import (
    # Loudness / Energy
    mean_amplitude,
    rms_energy,
    dynamic_range,
    peak_to_average_ratio,
    # Frequency / Spectral
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_flatness,
    zero_crossing_rate,
    fundamental_frequency,
    # Timbre / MFCCs
    mfcc_mean,
    mfcc_variance,
    # Rhythm / Music
    tempo,
    beat_strength,
    harmonic_ratio,
    # Quality
    compression_ratio,
)


TEST_AUDIO_DIR = Path(__file__).parent.parent / "test_data" / "audio"


def get_audio_path(name):
    """Get full path to a test audio file."""
    return TEST_AUDIO_DIR / f"{name}.wav"


def get_sorted_names(results, audios):
    """Extract audio names from sorted results."""
    return [audios[idx].stem for idx, *_ in results]


class TestMeanAmplitude:
    """Tests for mean_amplitude function."""

    @pytest.fixture
    def loudness_audios(self):
        return [
            get_audio_path("loudness_quiet"),
            get_audio_path("loudness_medium"),
            get_audio_path("loudness_loud"),
        ]

    def test_orders_quiet_before_loud(self, loudness_audios):
        """Quiet audio should come first when low_first=True."""
        results = mean_amplitude(loudness_audios, low_first=True)
        names = get_sorted_names(results, loudness_audios)

        assert names.index("loudness_quiet") < names.index("loudness_medium")
        assert names.index("loudness_medium") < names.index("loudness_loud")

    def test_orders_loud_before_quiet(self, loudness_audios):
        """Loud audio should come first when low_first=False."""
        results = mean_amplitude(loudness_audios, low_first=False)
        names = get_sorted_names(results, loudness_audios)

        assert names.index("loudness_loud") < names.index("loudness_medium")
        assert names.index("loudness_medium") < names.index("loudness_quiet")

    def test_returns_correct_structure(self, loudness_audios):
        """Results should be list of (index, amplitude) tuples."""
        results = mean_amplitude(loudness_audios)

        assert len(results) == 3
        for idx, amp in results:
            assert isinstance(idx, int)
            assert isinstance(amp, float)
            assert amp >= 0


class TestRmsEnergy:
    """Tests for rms_energy function."""

    @pytest.fixture
    def loudness_audios(self):
        return [
            get_audio_path("loudness_quiet"),
            get_audio_path("loudness_loud"),
        ]

    def test_orders_quiet_before_loud(self, loudness_audios):
        """Quiet audio should come first when low_first=True."""
        results = rms_energy(loudness_audios, low_first=True)
        names = get_sorted_names(results, loudness_audios)

        assert names[0] == "loudness_quiet"

    def test_loud_has_higher_energy(self, loudness_audios):
        """Loud audio should have higher RMS energy."""
        results = rms_energy(loudness_audios)
        scores = {loudness_audios[idx].stem: energy for idx, energy in results}

        assert scores["loudness_loud"] > scores["loudness_quiet"]


class TestDynamicRange:
    """Tests for dynamic_range function."""

    @pytest.fixture
    def dynamic_audios(self):
        return [
            get_audio_path("dynamic_low"),
            get_audio_path("dynamic_high"),
        ]

    def test_orders_low_dynamic_first(self, dynamic_audios):
        """Low dynamic range audio should come first when low_first=True."""
        results = dynamic_range(dynamic_audios, low_first=True)
        names = get_sorted_names(results, dynamic_audios)

        assert names[0] == "dynamic_low"

    def test_high_dynamic_has_larger_range(self, dynamic_audios):
        """High dynamic audio should have larger range."""
        results = dynamic_range(dynamic_audios)
        scores = {dynamic_audios[idx].stem: rng for idx, rng in results}

        assert scores["dynamic_high"] > scores["dynamic_low"]


class TestPeakToAverageRatio:
    """Tests for peak_to_average_ratio function."""

    @pytest.fixture
    def beat_audios(self):
        return [
            get_audio_path("beats_weak"),    # Sustained tone - low PAR
            get_audio_path("beats_strong"),  # Clicks - high PAR
        ]

    def test_sustained_has_lower_par(self, beat_audios):
        """Sustained tones should have lower peak-to-average ratio."""
        results = peak_to_average_ratio(beat_audios, low_first=True)
        names = get_sorted_names(results, beat_audios)

        assert names[0] == "beats_weak"


class TestSpectralCentroid:
    """Tests for spectral_centroid function."""

    @pytest.fixture
    def spectral_audios(self):
        return [
            get_audio_path("spectral_dark"),   # 100 Hz
            get_audio_path("spectral_mid"),    # 1000 Hz
            get_audio_path("spectral_bright"), # 4000 Hz
        ]

    def test_orders_dark_before_bright(self, spectral_audios):
        """Dark (low frequency) audio should come first when low_first=True."""
        results = spectral_centroid(spectral_audios, low_first=True)
        names = get_sorted_names(results, spectral_audios)

        assert names.index("spectral_dark") < names.index("spectral_mid")
        assert names.index("spectral_mid") < names.index("spectral_bright")

    def test_orders_bright_before_dark(self, spectral_audios):
        """Bright (high frequency) audio should come first when low_first=False."""
        results = spectral_centroid(spectral_audios, low_first=False)
        names = get_sorted_names(results, spectral_audios)

        assert names.index("spectral_bright") < names.index("spectral_mid")
        assert names.index("spectral_mid") < names.index("spectral_dark")


class TestSpectralBandwidth:
    """Tests for spectral_bandwidth function."""

    @pytest.fixture
    def bandwidth_audios(self):
        return [
            get_audio_path("bandwidth_narrow"),  # Single sine
            get_audio_path("bandwidth_broad"),   # Noise
        ]

    def test_orders_narrow_before_broad(self, bandwidth_audios):
        """Narrow-band audio should come first when low_first=True."""
        results = spectral_bandwidth(bandwidth_audios, low_first=True)
        names = get_sorted_names(results, bandwidth_audios)

        assert names[0] == "bandwidth_narrow"

    def test_noise_has_higher_bandwidth(self, bandwidth_audios):
        """Noise should have higher bandwidth than sine."""
        results = spectral_bandwidth(bandwidth_audios)
        scores = {bandwidth_audios[idx].stem: bw for idx, bw in results}

        assert scores["bandwidth_broad"] > scores["bandwidth_narrow"]


class TestSpectralRolloff:
    """Tests for spectral_rolloff function."""

    @pytest.fixture
    def spectral_audios(self):
        return [
            get_audio_path("spectral_dark"),
            get_audio_path("spectral_bright"),
        ]

    def test_dark_has_lower_rolloff(self, spectral_audios):
        """Dark audio should have lower rolloff frequency."""
        results = spectral_rolloff(spectral_audios, low_first=True)
        names = get_sorted_names(results, spectral_audios)

        assert names[0] == "spectral_dark"


class TestSpectralFlatness:
    """Tests for spectral_flatness function."""

    @pytest.fixture
    def flatness_audios(self):
        return [
            get_audio_path("flatness_tonal"),  # Pure sine
            get_audio_path("flatness_noisy"),  # White noise
        ]

    def test_orders_tonal_before_noisy(self, flatness_audios):
        """Tonal audio should come first when low_first=True."""
        results = spectral_flatness(flatness_audios, low_first=True)
        names = get_sorted_names(results, flatness_audios)

        assert names[0] == "flatness_tonal"

    def test_noise_has_higher_flatness(self, flatness_audios):
        """Noise should have higher spectral flatness than sine."""
        results = spectral_flatness(flatness_audios)
        scores = {flatness_audios[idx].stem: flat for idx, flat in results}

        assert scores["flatness_noisy"] > scores["flatness_tonal"]


class TestZeroCrossingRate:
    """Tests for zero_crossing_rate function."""

    @pytest.fixture
    def zcr_audios(self):
        return [
            get_audio_path("zcr_low"),   # Low frequency
            get_audio_path("zcr_high"),  # High frequency + noise
        ]

    def test_orders_low_zcr_first(self, zcr_audios):
        """Low ZCR audio should come first when low_first=True."""
        results = zero_crossing_rate(zcr_audios, low_first=True)
        names = get_sorted_names(results, zcr_audios)

        assert names[0] == "zcr_low"

    def test_high_freq_has_higher_zcr(self, zcr_audios):
        """High frequency audio should have higher ZCR."""
        results = zero_crossing_rate(zcr_audios)
        scores = {zcr_audios[idx].stem: zcr for idx, zcr in results}

        assert scores["zcr_high"] > scores["zcr_low"]


class TestFundamentalFrequency:
    """Tests for fundamental_frequency function."""

    @pytest.fixture
    def pitch_audios(self):
        return [
            get_audio_path("pitch_low"),   # 100 Hz
            get_audio_path("pitch_high"),  # 1000 Hz
        ]

    def test_orders_low_pitch_first(self, pitch_audios):
        """Low pitch audio should come first when low_first=True."""
        results = fundamental_frequency(pitch_audios, low_first=True)
        names = get_sorted_names(results, pitch_audios)

        assert names[0] == "pitch_low"

    def test_detects_pitch_correctly(self, pitch_audios):
        """Should detect pitch in correct range."""
        results = fundamental_frequency(pitch_audios)
        scores = {pitch_audios[idx].stem: f0 for idx, f0 in results}

        # Low pitch should be around 100 Hz
        assert 80 < scores["pitch_low"] < 150
        # High pitch should be around 1000 Hz
        assert 800 < scores["pitch_high"] < 1200


class TestMfccMean:
    """Tests for mfcc_mean function."""

    @pytest.fixture
    def loudness_audios(self):
        return [
            get_audio_path("loudness_quiet"),
            get_audio_path("loudness_loud"),
        ]

    def test_returns_correct_structure(self, loudness_audios):
        """Results should be (index, mfcc_mean) tuples."""
        results = mfcc_mean(loudness_audios)

        assert len(results) == 2
        for idx, val in results:
            assert isinstance(idx, int)
            assert isinstance(val, (int, float))


class TestMfccVariance:
    """Tests for mfcc_variance function."""

    @pytest.fixture
    def timbre_audios(self):
        return [
            get_audio_path("timbre_stable"),   # Constant sine
            get_audio_path("timbre_varying"),  # Frequency sweep
        ]

    def test_orders_stable_before_varying(self, timbre_audios):
        """Stable timbre should come first when low_first=True."""
        results = mfcc_variance(timbre_audios, low_first=True)
        names = get_sorted_names(results, timbre_audios)

        assert names[0] == "timbre_stable"

    def test_varying_has_higher_variance(self, timbre_audios):
        """Varying timbre should have higher MFCC variance."""
        results = mfcc_variance(timbre_audios)
        scores = {timbre_audios[idx].stem: var for idx, var in results}

        assert scores["timbre_varying"] > scores["timbre_stable"]


class TestTempo:
    """Tests for tempo function."""

    @pytest.fixture
    def tempo_audios(self):
        return [
            get_audio_path("tempo_slow"),  # ~60 BPM
            get_audio_path("tempo_fast"),  # ~180 BPM
        ]

    def test_orders_slow_before_fast(self, tempo_audios):
        """Slow tempo should come first when low_first=True."""
        results = tempo(tempo_audios, low_first=True)
        names = get_sorted_names(results, tempo_audios)

        assert names[0] == "tempo_slow"

    def test_fast_has_higher_bpm(self, tempo_audios):
        """Fast tempo audio should have higher BPM."""
        results = tempo(tempo_audios)
        scores = {tempo_audios[idx].stem: bpm for idx, bpm in results}

        assert scores["tempo_fast"] > scores["tempo_slow"]


class TestBeatStrength:
    """Tests for beat_strength function."""

    @pytest.fixture
    def beat_audios(self):
        return [
            get_audio_path("beats_weak"),    # Ambient sine
            get_audio_path("beats_strong"),  # Loud clicks
        ]

    def test_orders_weak_before_strong(self, beat_audios):
        """Weak beats should come first when low_first=True."""
        results = beat_strength(beat_audios, low_first=True)
        names = get_sorted_names(results, beat_audios)

        assert names[0] == "beats_weak"

    def test_strong_beats_have_higher_strength(self, beat_audios):
        """Strong beats should have higher onset strength."""
        results = beat_strength(beat_audios)
        scores = {beat_audios[idx].stem: strength for idx, strength in results}

        assert scores["beats_strong"] > scores["beats_weak"]


class TestHarmonicRatio:
    """Tests for harmonic_ratio function."""

    @pytest.fixture
    def harmonic_audios(self):
        return [
            get_audio_path("harmonic_percussive"),  # Clicks
            get_audio_path("harmonic_melodic"),     # Sustained tone
        ]

    def test_orders_percussive_before_melodic(self, harmonic_audios):
        """Percussive audio should come first when low_first=True."""
        results = harmonic_ratio(harmonic_audios, low_first=True)
        names = get_sorted_names(results, harmonic_audios)

        assert names[0] == "harmonic_percussive"

    def test_melodic_has_higher_harmonic_ratio(self, harmonic_audios):
        """Melodic audio should have higher harmonic ratio."""
        results = harmonic_ratio(harmonic_audios)
        scores = {harmonic_audios[idx].stem: ratio for idx, ratio in results}

        assert scores["harmonic_melodic"] > scores["harmonic_percussive"]


class TestCompressionRatio:
    """Tests for compression_ratio function."""

    @pytest.fixture
    def compress_audios(self):
        return [
            get_audio_path("compress_simple"),   # Sine wave
            get_audio_path("compress_complex"),  # Noise
        ]

    def test_simple_compresses_better(self, compress_audios):
        """Simple audio should have higher compression ratio."""
        results = compression_ratio(compress_audios, low_first=False)
        names = get_sorted_names(results, compress_audios)

        assert names[0] == "compress_simple"

    def test_orders_complex_first_when_low_first(self, compress_audios):
        """Complex audio should come first when low_first=True."""
        results = compression_ratio(compress_audios, low_first=True)
        names = get_sorted_names(results, compress_audios)

        assert names[0] == "compress_complex"


class TestEdgeCases:
    """Test edge cases for all audio sorters."""

    @pytest.fixture
    def single_audio(self):
        return [get_audio_path("loudness_medium")]

    def test_single_audio_mean_amplitude(self, single_audio):
        """mean_amplitude should work with single audio."""
        results = mean_amplitude(single_audio)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_single_audio_rms_energy(self, single_audio):
        """rms_energy should work with single audio."""
        results = rms_energy(single_audio)
        assert len(results) == 1

    def test_single_audio_spectral_centroid(self, single_audio):
        """spectral_centroid should work with single audio."""
        results = spectral_centroid(single_audio)
        assert len(results) == 1

    def test_single_audio_tempo(self, single_audio):
        """tempo should work with single audio."""
        results = tempo(single_audio)
        assert len(results) == 1

    def test_empty_list_mean_amplitude(self):
        """mean_amplitude should handle empty list."""
        assert mean_amplitude([]) == []

    def test_empty_list_rms_energy(self):
        """rms_energy should handle empty list."""
        assert rms_energy([]) == []

    def test_empty_list_spectral_centroid(self):
        """spectral_centroid should handle empty list."""
        assert spectral_centroid([]) == []

    def test_empty_list_spectral_bandwidth(self):
        """spectral_bandwidth should handle empty list."""
        assert spectral_bandwidth([]) == []

    def test_empty_list_spectral_flatness(self):
        """spectral_flatness should handle empty list."""
        assert spectral_flatness([]) == []

    def test_empty_list_zero_crossing_rate(self):
        """zero_crossing_rate should handle empty list."""
        assert zero_crossing_rate([]) == []

    def test_empty_list_mfcc_mean(self):
        """mfcc_mean should handle empty list."""
        assert mfcc_mean([]) == []

    def test_empty_list_tempo(self):
        """tempo should handle empty list."""
        assert tempo([]) == []

    def test_empty_list_beat_strength(self):
        """beat_strength should handle empty list."""
        assert beat_strength([]) == []

    def test_empty_list_harmonic_ratio(self):
        """harmonic_ratio should handle empty list."""
        assert harmonic_ratio([]) == []

    def test_empty_list_compression_ratio(self):
        """compression_ratio should handle empty list."""
        assert compression_ratio([]) == []
