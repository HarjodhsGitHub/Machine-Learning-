"""
AI Embedding pipeline using OpenL3 and FAISS for similarity search.
"""

import numpy as np
import librosa
import openl3
import faiss
import pickle
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """OpenL3-based embedding and FAISS indexing for music similarity."""

    def __init__(self, model_type: str = 'mel128', input_repr: str = 'mel256'):
        """
        Initialize OpenL3 embedding pipeline.

        Args:
            model_type: OpenL3 model architecture (mel128, mel256, music, etc.)
            input_repr: Input representation type
        """
        self.model_type = model_type
        self.input_repr = input_repr
        self.sr = 16000  # OpenL3 standard sample rate
        self.hop_size = 512
        self.model = None
        self.index = None
        self.metadata = []

        self._load_model()

    def _load_model(self):
        """Load pretrained OpenL3 model."""
        try:
            logger.info(f"Loading OpenL3 model: {self.model_type}")
            # Load model for the first time - will download if needed
            self.model = openl3.models.load_model(
                input_repr=self.input_repr,
                content_type='music',
                weights_dir=None
            )
            logger.info("OpenL3 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading OpenL3 model: {e}")
            raise

    def extract_embedding(self, file_path: str, use_jnorm: bool = True) -> np.ndarray:
        """
        Extract OpenL3 embeddings from audio file.

        Args:
            file_path: Path to audio file
            use_jnorm: Use L2 normalization (recommended for similarity search)

        Returns:
            Embedding vector (averaged across time)
        """
        try:
            logger.info(f"Extracting embedding: {file_path}")

            # Load audio at OpenL3's standard sample rate
            audio, sr = librosa.load(file_path, sr=self.sr, mono=True)

            # Extract embeddings
            embedding, timestamps = openl3.get_embeddings(
                audio,
                sr=self.sr,
                model=self.model,
                hop_size=self.hop_size
            )

            # Average embeddings across time
            embedding_mean = np.mean(embedding, axis=0)

            # L2 normalization
            if use_jnorm:
                embedding_mean = embedding_mean / (
                    np.linalg.norm(embedding_mean) + 1e-8
                )

            logger.info(
                f"Embedding extracted: shape={embedding_mean.shape}, "
                f"norm={np.linalg.norm(embedding_mean):.4f}"
            )
            return embedding_mean

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise

    def create_faiss_index(self, embedding_dim: int = 512) -> None:
        """
        Create FAISS index for similarity search.

        Args:
            embedding_dim: Dimension of embeddings
        """
        try:
            logger.info(f"Creating FAISS index (dimension: {embedding_dim})")

            # Use IndexFlatL2 for exact nearest neighbor search
            # Alternative: IndexIVFFlat for approximate search on large datasets
            self.index = faiss.IndexFlatL2(embedding_dim)

            logger.info("FAISS index created successfully")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise

    def add_to_index(
        self,
        embedding: np.ndarray,
        metadata: Dict,
        track_id: Optional[str] = None
    ) -> None:
        """
        Add embedding to FAISS index.

        Args:
            embedding: Embedding vector
            metadata: Track metadata (name, artist, bpm, key, etc.)
            track_id: Optional track identifier
        """
        try:
            if self.index is None:
                self._create_index(embedding.shape[0])

            # Ensure embedding is 2D for FAISS
            embedding_2d = np.array([embedding], dtype=np.float32)

            # Add to index
            self.index.add(embedding_2d)

            # Store metadata
            metadata_entry = {
                'track_id': track_id or len(self.metadata),
                'index_id': self.index.ntotal - 1,
                **metadata
            }
            self.metadata.append(metadata_entry)

            logger.info(
                f"Added to index: {metadata.get('title', 'Unknown')} "
                f"(total: {self.index.ntotal})"
            )
        except Exception as e:
            logger.error(f"Error adding to index: {e}")
            raise

    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        distance_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar tracks using FAISS.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            distance_threshold: Optional distance threshold

        Returns:
            List of similar tracks with metadata and distances
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Index is empty")
                return []

            # Ensure query is 2D
            query_2d = np.array([query_embedding], dtype=np.float32)

            # Search
            distances, indices = self.index.search(query_2d, k)

            # Build results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:  # Valid result
                    if distance_threshold is None or dist <= distance_threshold:
                        metadata = self.metadata[idx].copy()
                        metadata['distance'] = float(dist)
                        metadata['similarity'] = float(1 / (1 + dist))
                        results.append(metadata)

            logger.info(f"Found {len(results)} similar tracks")
            return results

        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []

    def save_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)

            # Save index
            faiss.write_index(self.index, index_path)
            logger.info(f"Index saved to {index_path}")

            # Save metadata
            if metadata_path is None:
                metadata_path = index_path.replace('.faiss', '_metadata.pkl')

            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Metadata saved to {metadata_path}")

        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise

    def load_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            # Load index
            self.index = faiss.read_index(index_path)
            logger.info(f"Index loaded from {index_path}")

            # Load metadata
            if metadata_path is None:
                metadata_path = index_path.replace('.faiss', '_metadata.pkl')

            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Metadata loaded from {metadata_path} ({len(self.metadata)} items)")

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def batch_add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict]
    ) -> None:
        """
        Add multiple embeddings to index at once.

        Args:
            embeddings: Array of embeddings (N x D)
            metadatas: List of metadata dictionaries
        """
        try:
            if self.index is None:
                self._create_index(embeddings.shape[1])

            # Ensure embeddings are float32
            embeddings = embeddings.astype(np.float32)

            # Add to index
            self.index.add(embeddings)

            # Store metadata
            for i, metadata in enumerate(metadatas):
                metadata_entry = {
                    'index_id': self.index.ntotal - len(embeddings) + i,
                    **metadata
                }
                self.metadata.append(metadata_entry)

            logger.info(f"Added {len(embeddings)} embeddings to index")

        except Exception as e:
            logger.error(f"Error batch adding embeddings: {e}")
            raise

    def _create_index(self, embedding_dim: int):
        """Internal helper to create index if it doesn't exist."""
        if self.index is None:
            self.create_faiss_index(embedding_dim)

    def get_index_stats(self) -> Dict:
        """Get statistics about current index."""
        if self.index is None:
            return {'ntotal': 0, 'dimension': 0}

        return {
            'ntotal': self.index.ntotal,
            'dimension': self.index.d,
            'metadata_entries': len(self.metadata)
        }


# Example usage and testing
if __name__ == '__main__':
    import tempfile
    import os

    # Create sample audio
    sr = 16000
    duration = 5  # seconds
    t = np.linspace(0, duration, sr * duration)
    # 440 Hz sine wave
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, 'test.wav')
        sf.write(audio_path, audio, sr)

        # Initialize pipeline
        pipeline = EmbeddingPipeline()

        # Extract embedding
        embedding = pipeline.extract_embedding(audio_path)
        print(f"Embedding shape: {embedding.shape}")

        # Create index and add embedding
        pipeline.create_faiss_index(embedding.shape[0])
        pipeline.add_to_index(
            embedding,
            {
                'title': 'Test Track',
                'artist': 'Test Artist',
                'bpm': 120,
                'key': 'C'
            }
        )

        # Search
        results = pipeline.search_similar(embedding, k=1)
        print(f"Search results: {results}")

        # Save index
        index_path = os.path.join(tmpdir, 'test_index.faiss')
        pipeline.save_index(index_path)

        # Load index
        pipeline2 = EmbeddingPipeline()
        pipeline2.load_index(index_path)
        print(f"Loaded index stats: {pipeline2.get_index_stats()}")
