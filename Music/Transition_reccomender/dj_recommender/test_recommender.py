"""
Unit tests for DJ Recommender components.
Run with: pytest test_recommender.py -v
"""

import sys
import os
import pytest
import numpy as np
import tempfile
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from audio_processor import AudioProcessor
from embedding_pipeline import EmbeddingPipeline
from transition_recommender import (
    TransitionRecommender,
    HarmonicMixer,
    TempoMatcher
)


class TestHarmonicMixer:
    """Tests for harmonic compatibility calculations."""

    def test_same_key_distance(self):
        """Same key should have zero distance."""
        assert HarmonicMixer.get_key_distance('C', 'C') == 0

    def test_adjacent_keys(self):
        """Adjacent keys should have distance 1."""
        assert HarmonicMixer.get_key_distance('C', 'C#') == 1
        assert HarmonicMixer.get_key_distance('C', 'B') == 1

    def test_harmonic_score_same_key(self):
        """Same key should score 1.0."""
        assert HarmonicMixer.calculate_harmonic_score('C', 'C') == 1.0

    def test_harmonic_score_tritone(self):
        """Tritone should score 0.0."""
        assert HarmonicMixer.calculate_harmonic_score('C', 'F#') == 0.0

    def test_compatible_keys(self):
        """Should return 5 compatible keys."""
        compatible = HarmonicMixer.get_compatible_keys('C', num_suggestions=5)
        assert len(compatible) == 5
        assert compatible[0][0] == 'C'  # Same key first
        assert compatible[0][1] == 1.0  # Score 1.0


class TestTempoMatcher:
    """Tests for tempo compatibility calculations."""

    def test_same_tempo(self):
        """Same tempo should score 1.0."""
        assert TempoMatcher.calculate_tempo_score(120, 120) == 1.0

    def test_halftime_tempo(self):
        """Halftime should have high score."""
        score = TempoMatcher.calculate_tempo_score(120, 60)
        assert score > 0.9

    def test_doubletime_tempo(self):
        """Double time should have high score."""
        score = TempoMatcher.calculate_tempo_score(120, 240)
        assert score > 0.9

    def test_five_percent_difference(self):
        """5% difference should score high."""
        score = TempoMatcher.calculate_tempo_score(120, 126)
        assert score > 0.9

    def test_twenty_percent_difference(self):
        """20% difference should score lower."""
        score = TempoMatcher.calculate_tempo_score(120, 144)
        assert 0.4 < score < 0.6

    def test_zero_bpm(self):
        """Zero BPM should return neutral score."""
        assert TempoMatcher.calculate_tempo_score(120, 0) == 0.5


class TestTransitionRecommender:
    """Tests for transition scoring."""

    def test_recommender_initialization(self):
        """Should initialize with correct weights."""
        rec = TransitionRecommender()
        # Weights should sum to 1.0 after normalization
        total = rec.harmonic_weight + rec.tempo_weight + rec.energy_weight + rec.embedding_weight
        assert abs(total - 1.0) < 0.001

    def test_transition_score_same_track(self):
        """Same track should have high score."""
        rec = TransitionRecommender()
        track = {
            'bpm': 120.0,
            'key': 'C',
            'rms_energy': 0.5
        }
        score = rec.calculate_transition_score(track, track, embedding_similarity=1.0)
        assert score.overall_score > 0.95

    def test_transition_score_components(self):
        """Score should have all components."""
        rec = TransitionRecommender()
        track1 = {'bpm': 120, 'key': 'C', 'rms_energy': 0.5}
        track2 = {'bpm': 125, 'key': 'G', 'rms_energy': 0.6}
        
        score = rec.calculate_transition_score(track1, track2, 0.7)
        
        assert 0 <= score.harmonic_score <= 1.0
        assert 0 <= score.tempo_score <= 1.0
        assert 0 <= score.energy_score <= 1.0
        assert 0 <= score.embedding_similarity <= 1.0
        assert 0 <= score.overall_score <= 1.0

    def test_recommendations_sorting(self):
        """Recommendations should be sorted by score."""
        rec = TransitionRecommender()
        current = {'bpm': 120, 'key': 'C', 'rms_energy': 0.5}
        
        candidates = [
            {'id': '1', 'bpm': 60, 'key': 'C#', 'rms_energy': 0.3},
            {'id': '2', 'bpm': 121, 'key': 'C', 'rms_energy': 0.51},
            {'id': '3', 'bpm': 240, 'key': 'F#', 'rms_energy': 0.7},
        ]
        
        recommendations = rec.recommend_transitions(
            current, candidates, top_k=3, min_score=0.0
        )
        
        assert len(recommendations) <= 3
        # Check sorting
        for i in range(len(recommendations) - 1):
            assert recommendations[i]['transition_score'] >= recommendations[i+1]['transition_score']


class TestAudioProcessor:
    """Tests for audio processing."""

    def create_test_audio(self, bpm=120, duration=5):
        """Create test audio file."""
        sr = 22050
        t = np.linspace(0, duration, sr * duration)
        # Simple sine wave
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            return f.name

    def test_load_audio(self):
        """Should load audio file correctly."""
        audio_file = self.create_test_audio()
        try:
            processor = AudioProcessor()
            y, sr = processor.load_audio(audio_file)
            
            assert isinstance(y, np.ndarray)
            assert sr == 22050
            assert len(y) > 0
        finally:
            os.unlink(audio_file)

    def test_extract_bpm(self):
        """Should extract BPM."""
        audio_file = self.create_test_audio()
        try:
            processor = AudioProcessor()
            y, sr = processor.load_audio(audio_file)
            bpm = processor.extract_bpm(y, sr)
            
            assert isinstance(bpm, float)
            assert bpm >= 0
        finally:
            os.unlink(audio_file)

    def test_extract_key(self):
        """Should extract key."""
        audio_file = self.create_test_audio()
        try:
            processor = AudioProcessor()
            y, sr = processor.load_audio(audio_file)
            key, confidence = processor.extract_key(y, sr)
            
            assert key in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            assert 0 <= confidence <= 1
        finally:
            os.unlink(audio_file)

    def test_full_processing(self):
        """Should process complete audio file."""
        audio_file = self.create_test_audio(duration=10)
        try:
            processor = AudioProcessor()
            features = processor.process_audio(audio_file)
            
            assert features.bpm > 0
            assert features.key is not None
            assert features.duration > 0
            assert features.rms_energy >= 0
            assert len(features.beat_frames) > 0
        finally:
            os.unlink(audio_file)


class TestEmbeddingPipeline:
    """Tests for embedding pipeline."""

    def create_test_audio(self, duration=5):
        """Create test audio file."""
        sr = 16000
        t = np.linspace(0, duration, sr * duration)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sr)
            return f.name

    @pytest.mark.skip(reason="Requires OpenL3 model download - slow on first run")
    def test_embedding_extraction(self):
        """Should extract embedding."""
        audio_file = self.create_test_audio()
        try:
            pipeline = EmbeddingPipeline()
            embedding = pipeline.extract_embedding(audio_file)
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 512  # OpenL3 default dimension
            assert abs(np.linalg.norm(embedding) - 1.0) < 0.01  # L2 normalized
        finally:
            os.unlink(audio_file)

    def test_faiss_index_creation(self):
        """Should create FAISS index."""
        pipeline = EmbeddingPipeline()
        pipeline.create_faiss_index(512)
        
        assert pipeline.index is not None
        assert pipeline.index.d == 512

    def test_index_persistence(self):
        """Should save and load index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test_index.faiss')
            
            # Create and save
            pipeline1 = EmbeddingPipeline()
            pipeline1.create_faiss_index(512)
            
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            
            pipeline1.add_to_index(emb, {'title': 'Test'}, 'track_1')
            pipeline1.save_index(index_path)
            
            # Load
            pipeline2 = EmbeddingPipeline()
            pipeline2.load_index(index_path)
            
            stats = pipeline2.get_index_stats()
            assert stats['ntotal'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
