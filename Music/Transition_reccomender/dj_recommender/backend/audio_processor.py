"""
Audio processing module for BPM detection, key detection, and beat grid extraction.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MUSICAL_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    bpm: float
    key: str
    key_confidence: float
    beat_frames: np.ndarray
    onset_frames: np.ndarray
    spectral_centroid: float
    mfcc_mean: np.ndarray
    zero_crossing_rate: float
    rms_energy: float
    chroma_vector: np.ndarray
    duration: float
    sample_rate: int


class AudioProcessor:
    """Main audio processing class for music analysis."""

    def __init__(self, sr: int = 22050, hop_length: int = 512):
        """
        Initialize AudioProcessor.

        Args:
            sr: Sample rate for processing
            hop_length: Hop length for feature extraction
        """
        self.sr = sr
        self.hop_length = hop_length

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with resampling to target sample rate."""
        try:
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            logger.info(f"Loaded audio: {file_path} (duration: {len(y)/sr:.2f}s)")
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise

    def extract_bpm(self, y: np.ndarray, sr: int) -> float:
        """
        Extract BPM using librosa's tempo detection.
        Uses onset detection + dynamic programming for robust BPM estimation.
        """
        try:
            # Compute onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Estimate tempo
            tempo, beats = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=self.hop_length
            )
            logger.info(f"Detected BPM: {tempo:.2f}")
            return float(tempo)
        except Exception as e:
            logger.error(f"Error extracting BPM: {e}")
            return 0.0

    def extract_key(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """
        Extract musical key using chromagram and energy distribution.
        Uses Krumhansl-Schmuckler key-finding algorithm via chroma features.
        """
        try:
            # Compute constant-Q chroma features
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            # Average chroma over time
            chroma_mean = np.mean(chroma, axis=1)

            # Camelot wheel major keys and their chroma distributions
            major_profiles = {
                'C': np.array([1.0, 0.0, 0.08, 0.0, 0.08, 0.08, 0.0, 1.0, 0.0, 0.52, 0.0, 0.4]),
                'C#': np.array([0.4, 1.0, 0.0, 0.08, 0.0, 0.08, 0.08, 0.0, 1.0, 0.0, 0.52, 0.0]),
                'D': np.array([0.0, 0.4, 1.0, 0.0, 0.08, 0.0, 0.08, 0.08, 0.0, 1.0, 0.0, 0.52]),
                'D#': np.array([0.52, 0.0, 0.4, 1.0, 0.0, 0.08, 0.0, 0.08, 0.08, 0.0, 1.0, 0.0]),
                'E': np.array([0.0, 0.52, 0.0, 0.4, 1.0, 0.0, 0.08, 0.0, 0.08, 0.08, 0.0, 1.0]),
                'F': np.array([1.0, 0.0, 0.52, 0.0, 0.4, 1.0, 0.0, 0.08, 0.0, 0.08, 0.08, 0.0]),
                'F#': np.array([0.0, 1.0, 0.0, 0.52, 0.0, 0.4, 1.0, 0.0, 0.08, 0.0, 0.08, 0.08]),
                'G': np.array([0.08, 0.0, 1.0, 0.0, 0.52, 0.0, 0.4, 1.0, 0.0, 0.08, 0.0, 0.08]),
                'G#': np.array([0.08, 0.08, 0.0, 1.0, 0.0, 0.52, 0.0, 0.4, 1.0, 0.0, 0.08, 0.0]),
                'A': np.array([0.0, 0.08, 0.08, 0.0, 1.0, 0.0, 0.52, 0.0, 0.4, 1.0, 0.0, 0.08]),
                'A#': np.array([0.08, 0.0, 0.08, 0.08, 0.0, 1.0, 0.0, 0.52, 0.0, 0.4, 1.0, 0.0]),
                'B': np.array([0.0, 0.08, 0.0, 0.08, 0.08, 0.0, 1.0, 0.0, 0.52, 0.0, 0.4, 1.0]),
            }

            # Calculate correlation with each key profile
            correlations = {}
            for note, profile in major_profiles.items():
                # Normalize vectors
                profile_norm = profile / np.linalg.norm(profile)
                chroma_norm = chroma_mean / np.linalg.norm(chroma_mean)
                correlations[note] = np.dot(profile_norm, chroma_norm)

            # Find best matching key
            best_key = max(correlations, key=correlations.get)
            confidence = (correlations[best_key] + 1) / 2  # Normalize to 0-1
            
            logger.info(f"Detected key: {best_key} (confidence: {confidence:.2f})")
            return best_key, float(confidence)
        except Exception as e:
            logger.error(f"Error extracting key: {e}")
            return 'C', 0.5

    def extract_beat_grid(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract beat positions in frames."""
        try:
            _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            logger.info(f"Extracted {len(beats)} beat frames")
            return beats
        except Exception as e:
            logger.error(f"Error extracting beat grid: {e}")
            return np.array([])

    def extract_onsets(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract onset times (note attack times)."""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
            onsets = librosa.onset.onset_detect(
                onset_strength=onset_env,
                sr=sr,
                hop_length=self.hop_length,
                backtrack=True
            )
            logger.info(f"Detected {len(onsets)} onsets")
            return onsets
        except Exception as e:
            logger.error(f"Error extracting onsets: {e}")
            return np.array([])

    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract spectral features: MFCC, spectral centroid, zero-crossing rate."""
        try:
            # Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)

            # MFCCs (Mel-Frequency Cepstral Coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)

            # Zero-Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y=y)[0]
            zcr_mean = np.mean(zcr)

            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)

            logger.info("Extracted spectral features")
            return {
                'spectral_centroid': spectral_centroid_mean,
                'mfcc_mean': mfcc_mean,
                'zero_crossing_rate': zcr_mean,
                'rms_energy': rms_mean
            }
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return {}

    def extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract chroma vector for harmonic content analysis."""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            logger.info("Extracted chroma vector")
            return chroma_mean
        except Exception as e:
            logger.error(f"Error extracting chroma: {e}")
            return np.zeros(12)

    def process_audio(self, file_path: str) -> AudioFeatures:
        """
        Complete audio processing pipeline.
        Extracts all features and returns AudioFeatures object.
        """
        logger.info(f"Processing audio: {file_path}")

        # Load audio
        y, sr = self.load_audio(file_path)
        duration = len(y) / sr

        # Extract features
        bpm = self.extract_bpm(y, sr)
        key, key_confidence = self.extract_key(y, sr)
        beat_frames = self.extract_beat_grid(y, sr)
        onset_frames = self.extract_onsets(y, sr)

        # Extract spectral features
        spectral_features = self.extract_spectral_features(y, sr)

        # Extract chroma
        chroma_vector = self.extract_chroma(y, sr)

        features = AudioFeatures(
            bpm=bpm,
            key=key,
            key_confidence=key_confidence,
            beat_frames=beat_frames,
            onset_frames=onset_frames,
            spectral_centroid=spectral_features.get('spectral_centroid', 0.0),
            mfcc_mean=spectral_features.get('mfcc_mean', np.zeros(13)),
            zero_crossing_rate=spectral_features.get('zero_crossing_rate', 0.0),
            rms_energy=spectral_features.get('rms_energy', 0.0),
            chroma_vector=chroma_vector,
            duration=duration,
            sample_rate=sr
        )

        logger.info(f"Audio processing complete: BPM={bpm:.2f}, Key={key}")
        return features

    def resample_audio(self, file_path: str, target_sr: int = 22050) -> str:
        """
        Resample audio to target sample rate and save.
        Useful for standardizing library audio.
        """
        try:
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)
            output_path = file_path.replace('.wav', '_resampled.wav')
            sf.write(output_path, y, target_sr)
            logger.info(f"Resampled audio saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            raise
