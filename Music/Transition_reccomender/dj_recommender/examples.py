"""
Example script demonstrating the DJ Recommender system programmatically.
Run this to test all components without the web interface.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from audio_processor import AudioProcessor, AudioFeatures
from embedding_pipeline import EmbeddingPipeline
from transition_recommender import (
    TransitionRecommender,
    HarmonicMixer,
    TempoMatcher
)
import numpy as np
import soundfile as sf


def create_sample_audio(filename: str, bpm: float = 120, duration: float = 10):
    """Create a sample sine wave audio file for testing."""
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create frequency based on BPM
    freq = bpm / 60  # Hz
    
    # Generate tone
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    sf.write(filename, audio, sr)
    print(f"✓ Created sample audio: {filename}")
    return filename


def example_1_audio_processing():
    """Example 1: Audio feature extraction."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Audio Processing & Feature Extraction")
    print("="*60)
    
    # Create sample file
    sample_file = "sample_track.wav"
    create_sample_audio(sample_file, bpm=128, duration=30)
    
    # Process audio
    processor = AudioProcessor()
    features = processor.process_audio(sample_file)
    
    print(f"\n📊 Audio Analysis Results:")
    print(f"   BPM: {features.bpm:.2f}")
    print(f"   Key: {features.key} (confidence: {features.key_confidence:.2f})")
    print(f"   Duration: {features.duration:.2f}s")
    print(f"   RMS Energy: {features.rms_energy:.4f}")
    print(f"   Spectral Centroid: {features.spectral_centroid:.2f} Hz")
    print(f"   Zero-Crossing Rate: {features.zero_crossing_rate:.4f}")
    print(f"   Beat Frames: {len(features.beat_frames)} detected")
    print(f"   Onset Frames: {len(features.onset_frames)} detected")
    
    # Cleanup
    os.remove(sample_file)
    return features


def example_2_harmonic_mixing():
    """Example 2: Harmonic mixing rules."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Harmonic Compatibility Analysis")
    print("="*60)
    
    keys = ['C', 'G', 'F', 'A#']
    
    print("\n🎸 Key Compatibility Matrix:")
    print("       ", end="")
    for k in keys:
        print(f"{k:>8}", end="")
    print()
    
    for key1 in keys:
        print(f"{key1:>5}:", end="")
        for key2 in keys:
            score = HarmonicMixer.calculate_harmonic_score(key1, key2)
            print(f"{score:>8.2f}", end="")
        print()
    
    # Get compatible keys for C major
    print("\n🔑 Compatible keys for C major:")
    compatible = HarmonicMixer.get_compatible_keys('C', num_suggestions=6)
    for key, score in compatible:
        bar = "█" * int(score * 20)
        print(f"   {key}: {bar} {score:.2f}")


def example_3_tempo_matching():
    """Example 3: Tempo compatibility."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Tempo Matching Analysis")
    print("="*60)
    
    base_bpm = 120
    test_bpms = [60, 120, 122, 124, 240]  # halftime, same, close, close, doubletime
    
    print(f"\n⏱️ Tempo compatibility from {base_bpm} BPM:")
    for bpm in test_bpms:
        score = TempoMatcher.calculate_tempo_score(base_bpm, bpm)
        ratio = bpm / base_bpm
        bar = "█" * int(score * 20)
        print(f"   {bpm:>3} BPM (ratio {ratio:.2f}): {bar} {score:.2f}")


def example_4_transition_scoring():
    """Example 4: Full transition recommendation scoring."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Transition Scoring")
    print("="*60)
    
    # Create sample tracks
    track_a = {
        'id': 'track_a',
        'title': 'Deep House Vibes',
        'bpm': 125.0,
        'key': 'A',
        'rms_energy': 0.65
    }
    
    track_b = {
        'id': 'track_b',
        'title': 'Tech House Drop',
        'bpm': 128.0,
        'key': 'E',
        'rms_energy': 0.72
    }
    
    track_c = {
        'id': 'track_c',
        'title': 'Progressive Chill',
        'bpm': 124.0,
        'key': 'A',
        'rms_energy': 0.60
    }
    
    # Initialize recommender
    recommender = TransitionRecommender()
    
    # Score transitions from track_a
    candidates = [track_b, track_c]
    recommendations = recommender.recommend_transitions(
        track_a, candidates, top_k=10, min_score=0.3
    )
    
    print(f"\n🎯 Recommendations from '{track_a['title']}':")
    print(f"   {track_a['key']} key • {track_a['bpm']:.0f} BPM • {track_a['rms_energy']:.2f} energy\n")
    
    for rec in recommendations:
        print(f"   ➜ {rec['track']['title']}")
        print(f"      Score: {rec['transition_score']:.3f} " +
              f"(Harmonic: {rec['harmonic_score']:.2f}, " +
              f"Tempo: {rec['tempo_score']:.2f}, " +
              f"Energy: {rec['energy_score']:.2f})")
        print(f"      {rec['explanation']}")
        print()


def example_5_embedding_pipeline():
    """Example 5: Embedding extraction and indexing."""
    print("\n" + "="*60)
    print("EXAMPLE 5: AI Embeddings & FAISS Indexing")
    print("="*60)
    
    try:
        # Create sample files
        print("\n🎵 Creating sample audio files...")
        file1 = create_sample_audio("track1_low.wav", bpm=120, duration=10)
        file2 = create_sample_audio("track2_mid.wav", bpm=125, duration=10)
        file3 = create_sample_audio("track3_high.wav", bpm=130, duration=10)
        
        # Initialize pipeline
        print("\n🔄 Initializing OpenL3 embedding pipeline...")
        pipeline = EmbeddingPipeline()
        
        # Extract embeddings
        print("\n📊 Extracting embeddings...")
        emb1 = pipeline.extract_embedding(file1)
        emb2 = pipeline.extract_embedding(file2)
        emb3 = pipeline.extract_embedding(file3)
        
        print(f"   Embedding shape: {emb1.shape}")
        print(f"   Norm: {np.linalg.norm(emb1):.4f}")
        
        # Create index
        print("\n🗂️ Creating FAISS index...")
        pipeline.create_faiss_index(emb1.shape[0])
        
        # Add embeddings
        print("📍 Adding embeddings to index...")
        pipeline.add_to_index(emb1, {'title': 'Low Intensity', 'artist': 'Artist A'}, 'track1')
        pipeline.add_to_index(emb2, {'title': 'Mid Intensity', 'artist': 'Artist B'}, 'track2')
        pipeline.add_to_index(emb3, {'title': 'High Intensity', 'artist': 'Artist C'}, 'track3')
        
        # Search
        print("\n🔍 Searching for similar tracks to Track 1...")
        results = pipeline.search_similar(emb1, k=3)
        for i, result in enumerate(results):
            print(f"   {i+1}. {result['title']} - Similarity: {result['similarity']:.4f}")
        
        # Save index
        print("\n💾 Saving index...")
        pipeline.save_index('demo_index.faiss')
        
        # Load index
        print("📂 Loading index...")
        pipeline2 = EmbeddingPipeline()
        pipeline2.load_index('demo_index.faiss')
        stats = pipeline2.get_index_stats()
        print(f"   Index loaded: {stats['ntotal']} items, {stats['dimension']} dimensions")
        
        # Cleanup
        for f in [file1, file2, file3, 'demo_index.faiss', 'demo_index_metadata.pkl']:
            if os.path.exists(f):
                os.remove(f)
        
    except Exception as e:
        print(f"\n⚠️ OpenL3 model download required on first run (can be slow):")
        print(f"   Error: {e}")
        print(f"   The model will be downloaded automatically when needed (~600MB)")


def main():
    """Run all examples."""
    print("\n" + "🎵 "*20)
    print("DJ TRANSITION RECOMMENDER - EXAMPLE DEMONSTRATIONS")
    print("🎵 "*20)
    
    try:
        # Run examples
        example_1_audio_processing()
        example_2_harmonic_mixing()
        example_3_tempo_matching()
        example_4_transition_scoring()
        example_5_embedding_pipeline()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("="*60)
        print("\n📖 Next steps:")
        print("   1. Run the web app: python backend/main.py")
        print("   2. Upload audio files via http://localhost:8000")
        print("   3. Get AI-powered transition recommendations")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
