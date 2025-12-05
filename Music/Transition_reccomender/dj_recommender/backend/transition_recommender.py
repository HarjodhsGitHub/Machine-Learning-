"""
DJ Transition Recommender using harmonic rules and embedding similarity.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransitionScore:
    """Container for transition score components."""
    harmonic_score: float  # 0-1: How well keys mix
    energy_score: float    # 0-1: Energy progression
    tempo_score: float     # 0-1: How compatible tempos are
    embedding_similarity: float  # 0-1: AI-learned similarity
    overall_score: float   # 0-1: Weighted combination
    explanation: str       # Human-readable reason


class HarmonicMixer:
    """Harmonic mixing rules for DJ transitions."""

    # Camelot wheel positions (12 o'clock positions for each key)
    CAMELOT_WHEEL = {
        'B': 1, 'F#': 2, 'C#': 3, 'G#': 4, 'D#': 5, 'A#': 6,
        'F': 7, 'C': 8, 'G': 9, 'D': 10, 'A': 11, 'E': 12
    }

    # Harmonic compatibility rules
    # Same key or adjacent keys (±1 semitone) harmonize well
    @staticmethod
    def get_key_distance(key1: str, key2: str) -> int:
        """
        Get semitone distance between two keys.
        Used to determine harmonic compatibility.
        """
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        try:
            idx1 = notes.index(key1)
            idx2 = notes.index(key2)
            distance = abs(idx2 - idx1)
            # Prefer shortest path
            distance = min(distance, 12 - distance)
            return distance
        except ValueError:
            return 6  # Neutral if key not found

    @staticmethod
    def calculate_harmonic_score(key1: str, key2: str) -> float:
        """
        Calculate harmonic compatibility score (0-1).
        Same key = 1.0, adjacent = 0.8, minor third = 0.6, tritone = 0.0
        """
        distance = HarmonicMixer.get_key_distance(key1, key2)

        harmonic_scores = {
            0: 1.0,   # Same key (perfect match)
            1: 0.9,   # Half step (very compatible)
            2: 0.8,   # Whole step (compatible)
            3: 0.7,   # Minor third (good)
            4: 0.6,   # Major third
            5: 0.4,   # Perfect fourth
            6: 0.0,   # Tritone (avoid)
        }

        return harmonic_scores.get(distance, 0.5)

    @staticmethod
    def get_compatible_keys(key: str, num_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get list of compatible keys sorted by harmonic score."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        scores = []
        for note in notes:
            score = HarmonicMixer.calculate_harmonic_score(key, note)
            scores.append((note, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:num_suggestions]


class TempoMatcher:
    """Tempo compatibility analysis for smooth transitions."""

    @staticmethod
    def calculate_tempo_score(bpm1: float, bpm2: float) -> float:
        """
        Calculate tempo compatibility score.
        - Same BPM or halftime/double time: high score
        - Within 10% variation: good score
        - Beyond 20% variation: low score
        """
        if bpm1 == 0 or bpm2 == 0:
            return 0.5  # Neutral if BPM not detected

        ratio = max(bpm1, bpm2) / min(bpm1, bpm2)

        # Perfect ratios (halftime, double time, etc.)
        perfect_ratios = [0.5, 1.0, 2.0]
        for ratio_target in perfect_ratios:
            if abs(ratio - ratio_target) < 0.05:  # Within 5% tolerance
                return 1.0

        # Check percentage difference
        percentage_diff = abs(bpm2 - bpm1) / max(bpm1, bpm2)

        if percentage_diff <= 0.05:  # Within 5%
            return 0.95
        elif percentage_diff <= 0.1:  # Within 10%
            return 0.85
        elif percentage_diff <= 0.15:  # Within 15%
            return 0.7
        elif percentage_diff <= 0.2:  # Within 20%
            return 0.5
        else:
            return max(0.2, 1.0 - (percentage_diff * 2))

    @staticmethod
    def get_halftime_tempo(bpm: float) -> float:
        """Calculate halftime tempo (half the original)."""
        return bpm / 2

    @staticmethod
    def get_doubletime_tempo(bpm: float) -> float:
        """Calculate double-time tempo (double the original)."""
        return bpm * 2


class TransitionRecommender:
    """Main recommendation engine combining harmonic, tempo, and embedding similarity."""

    def __init__(
        self,
        harmonic_weight: float = 0.35,
        tempo_weight: float = 0.25,
        energy_weight: float = 0.15,
        embedding_weight: float = 0.25
    ):
        """
        Initialize recommender with score weights.

        Args:
            harmonic_weight: Weight for harmonic compatibility
            tempo_weight: Weight for tempo compatibility
            energy_weight: Weight for energy progression
            embedding_weight: Weight for AI embedding similarity
        """
        total = harmonic_weight + tempo_weight + energy_weight + embedding_weight
        self.harmonic_weight = harmonic_weight / total
        self.tempo_weight = tempo_weight / total
        self.energy_weight = energy_weight / total
        self.embedding_weight = embedding_weight / total

        logger.info(
            f"Initialized recommender with weights: "
            f"harmonic={self.harmonic_weight:.2f}, "
            f"tempo={self.tempo_weight:.2f}, "
            f"energy={self.energy_weight:.2f}, "
            f"embedding={self.embedding_weight:.2f}"
        )

    def calculate_transition_score(
        self,
        current_track: Dict,
        candidate_track: Dict,
        embedding_similarity: float = 0.5
    ) -> TransitionScore:
        """
        Calculate comprehensive transition score.

        Args:
            current_track: Current track metadata (bpm, key, energy, etc.)
            candidate_track: Candidate track metadata
            embedding_similarity: Similarity from embedding model (0-1)

        Returns:
            TransitionScore with all components
        """
        # Harmonic score
        current_key = current_track.get('key', 'C')
        candidate_key = candidate_track.get('key', 'C')
        harmonic_score = HarmonicMixer.calculate_harmonic_score(
            current_key, candidate_key
        )

        # Tempo score
        current_bpm = current_track.get('bpm', 120.0)
        candidate_bpm = candidate_track.get('bpm', 120.0)
        tempo_score = TempoMatcher.calculate_tempo_score(current_bpm, candidate_bpm)

        # Energy score (prefer increasing or similar energy)
        current_energy = current_track.get('rms_energy', 0.5)
        candidate_energy = candidate_track.get('rms_energy', 0.5)
        
        if candidate_energy >= current_energy:
            # Increasing or same energy (good for building)
            energy_score = min(1.0, (candidate_energy / (current_energy + 0.01)))
        else:
            # Decreasing energy (less ideal but acceptable)
            energy_score = max(0.3, candidate_energy / (current_energy + 0.01))

        # Embedding similarity (already 0-1)
        embedding_score = embedding_similarity

        # Calculate weighted overall score
        overall_score = (
            self.harmonic_weight * harmonic_score +
            self.tempo_weight * tempo_score +
            self.energy_weight * energy_score +
            self.embedding_weight * embedding_score
        )

        # Generate explanation
        explanation = self._generate_explanation(
            current_key, candidate_key, current_bpm, candidate_bpm,
            harmonic_score, tempo_score, energy_score, embedding_score
        )

        return TransitionScore(
            harmonic_score=harmonic_score,
            energy_score=energy_score,
            tempo_score=tempo_score,
            embedding_similarity=embedding_score,
            overall_score=overall_score,
            explanation=explanation
        )

    def _generate_explanation(
        self,
        current_key: str,
        candidate_key: str,
        current_bpm: float,
        candidate_bpm: float,
        harmonic_score: float,
        tempo_score: float,
        energy_score: float,
        embedding_score: float
    ) -> str:
        """Generate human-readable transition explanation."""
        parts = []

        # Harmonic explanation
        if harmonic_score > 0.8:
            parts.append(f"✓ Harmonic: {current_key} → {candidate_key} (excellent key match)")
        elif harmonic_score > 0.5:
            parts.append(f"◐ Harmonic: {current_key} → {candidate_key} (good harmonic compatibility)")
        else:
            parts.append(f"✗ Harmonic: {current_key} → {candidate_key} (different key - beatmix required)")

        # Tempo explanation
        if tempo_score > 0.9:
            parts.append(f"✓ Tempo: {current_bpm:.0f} → {candidate_bpm:.0f} BPM (perfect match)")
        elif tempo_score > 0.7:
            parts.append(f"◐ Tempo: {current_bpm:.0f} → {candidate_bpm:.0f} BPM (compatible)")
        else:
            parts.append(f"✗ Tempo: {current_bpm:.0f} → {candidate_bpm:.0f} BPM (requires beatmatching)")

        # Energy explanation
        if energy_score > 0.8:
            parts.append("✓ Energy: Rising energy (great build)")
        elif energy_score > 0.5:
            parts.append("◐ Energy: Similar energy level")
        else:
            parts.append("◐ Energy: Dropping energy (good for cooldown)")

        # Embedding explanation
        if embedding_score > 0.8:
            parts.append(f"✓ Vibe: Very similar sonic character (similarity: {embedding_score:.2f})")
        elif embedding_score > 0.6:
            parts.append(f"◐ Vibe: Similar sonic character (similarity: {embedding_score:.2f})")
        else:
            parts.append(f"◐ Vibe: Different sonic character (similarity: {embedding_score:.2f})")

        return " | ".join(parts)

    def recommend_transitions(
        self,
        current_track: Dict,
        candidate_tracks: List[Dict],
        top_k: int = 10,
        min_score: float = 0.5
    ) -> List[Dict]:
        """
        Get ranked list of recommended transitions.

        Args:
            current_track: Current playing track
            candidate_tracks: List of candidate tracks to recommend from
            top_k: Number of recommendations to return
            min_score: Minimum score threshold

        Returns:
            Sorted list of recommended tracks with scores and explanations
        """
        recommendations = []

        for candidate in candidate_tracks:
            # Skip if same track
            if candidate.get('id') == current_track.get('id'):
                continue

            # Extract embedding similarity if available
            embedding_sim = candidate.get('embedding_similarity', 0.5)

            # Calculate transition score
            score_obj = self.calculate_transition_score(
                current_track, candidate, embedding_sim
            )

            if score_obj.overall_score >= min_score:
                recommendation = {
                    'track': candidate,
                    'transition_score': score_obj.overall_score,
                    'harmonic_score': score_obj.harmonic_score,
                    'tempo_score': score_obj.tempo_score,
                    'energy_score': score_obj.energy_score,
                    'embedding_similarity': score_obj.embedding_similarity,
                    'explanation': score_obj.explanation
                }
                recommendations.append(recommendation)

        # Sort by overall score descending
        recommendations.sort(
            key=lambda x: x['transition_score'],
            reverse=True
        )

        return recommendations[:top_k]

    def get_mix_strategy(
        self,
        current_track: Dict,
        candidate_track: Dict,
        transition_score: TransitionScore
    ) -> Dict[str, str]:
        """
        Suggest mixing strategy based on transition analysis.

        Returns:
            Dictionary with mixing tips
        """
        strategy = {}

        # EQ suggestions
        if transition_score.harmonic_score < 0.5:
            strategy['eq'] = (
                "Use high-pass filter on bass of incoming track "
                "to avoid frequency clash. Gradually transition EQ over 8-16 bars."
            )
        else:
            strategy['eq'] = "Keys are compatible - minimal EQ adjustment needed"

        # Cue strategy
        if transition_score.tempo_score > 0.9:
            strategy['cueing'] = "Cue incoming track at 0.0 (start). Tempos match perfectly."
        elif transition_score.tempo_score > 0.7:
            strategy['cueing'] = (
                "Find a strong kick/beat in incoming track. "
                "Beatmatch before mixing to locked position."
            )
        else:
            strategy['cueing'] = (
                "Significant tempo difference detected. Consider loop-based transition "
                "or extend the current track during mixing."
            )

        # Fader strategy
        if transition_score.energy_score > 0.7:
            strategy['fader'] = "Energy rising - smooth crossfader transition recommended"
        else:
            strategy['fader'] = "Energy similar/dropping - use longer crossfader for smoother mix"

        # Overall recommendation
        overall_score = transition_score.overall_score
        if overall_score > 0.85:
            strategy['difficulty'] = 'Easy - Professional-level match'
        elif overall_score > 0.7:
            strategy['difficulty'] = 'Moderate - Good transition with minor adjustments'
        elif overall_score > 0.5:
            strategy['difficulty'] = 'Challenging - Requires beatmatching and EQ work'
        else:
            strategy['difficulty'] = 'Very challenging - Consider alternative tracks'

        return strategy


# Example usage
if __name__ == '__main__':
    # Example tracks
    track1 = {
        'id': '1',
        'title': 'Track A',
        'bpm': 120.0,
        'key': 'C',
        'rms_energy': 0.6
    }

    track2 = {
        'id': '2',
        'title': 'Track B',
        'bpm': 124.0,
        'key': 'G',
        'rms_energy': 0.7
    }

    track3 = {
        'id': '3',
        'title': 'Track C',
        'bpm': 121.0,
        'key': 'C',
        'rms_energy': 0.75
    }

    # Initialize recommender
    recommender = TransitionRecommender()

    # Get recommendations
    candidates = [track2, track3]
    recommendations = recommender.recommend_transitions(track1, candidates)

    print("Recommended transitions:")
    for rec in recommendations:
        print(f"\n{rec['track']['title']}")
        print(f"  Overall Score: {rec['transition_score']:.2f}")
        print(f"  {rec['explanation']}")
        strategy = recommender.get_mix_strategy(track1, rec['track'], 
            TransitionScore(
                rec['harmonic_score'],
                rec['energy_score'],
                rec['tempo_score'],
                rec['embedding_similarity'],
                rec['transition_score'],
                rec['explanation']
            )
        )
        print(f"  Strategy: {strategy['difficulty']}")
