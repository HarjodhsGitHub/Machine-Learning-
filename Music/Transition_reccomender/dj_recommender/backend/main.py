"""
FastAPI backend server for DJ Transition Recommender.
Handles audio uploads, analysis, and recommendation requests.
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import shutil
import logging
from pathlib import Path
import json
from datetime import datetime
import asyncio

from audio_processor import AudioProcessor, AudioFeatures
from embedding_pipeline import EmbeddingPipeline
from transition_recommender import TransitionRecommender, TransitionScore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="DJ Transition Recommender API",
    description="AI-powered DJ mixing assistant with harmonic and embedding-based recommendations",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = os.getenv('UPLOAD_DIR', './uploads')
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', './data/music_index.faiss')
LIBRARY_STORAGE = os.getenv('LIBRARY_STORAGE', './data/library.json')

# Create directories
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

# Initialize components
audio_processor = AudioProcessor(sr=22050, hop_length=512)
embedding_pipeline = EmbeddingPipeline(input_repr='mel256')
transition_recommender = TransitionRecommender()

# Library storage (in-memory, persists to JSON)
music_library: Dict[str, dict] = {}


# ==================== Request/Response Models ====================

class TrackMetadata(BaseModel):
    """Track metadata schema."""
    id: str
    title: str
    artist: str
    filename: str
    bpm: float
    key: str
    key_confidence: float
    duration: float
    rms_energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    uploaded_at: str


class AnalysisResult(BaseModel):
    """Audio analysis result schema."""
    track_id: str
    filename: str
    duration: float
    bpm: float
    key: str
    key_confidence: float
    harmonic_compatible_keys: List[tuple]
    spectral_centroid: float
    rms_energy: float
    zero_crossing_rate: float
    analysis_timestamp: str


class RecommendationRequest(BaseModel):
    """Transition recommendation request."""
    track_id: str
    top_k: int = 10
    min_score: float = 0.5


class RecommendationResponse(BaseModel):
    """Transition recommendation response."""
    current_track: Dict
    recommendations: List[Dict]
    generation_timestamp: str


class TrackInfo(BaseModel):
    """Track info request."""
    title: str
    artist: str


# ==================== Utility Functions ====================

def save_library():
    """Persist library to JSON."""
    try:
        with open(LIBRARY_STORAGE, 'w') as f:
            json.dump(music_library, f, indent=2)
        logger.info("Library saved to disk")
    except Exception as e:
        logger.error(f"Error saving library: {e}")


def load_library():
    """Load library from JSON."""
    global music_library
    try:
        if os.path.exists(LIBRARY_STORAGE):
            with open(LIBRARY_STORAGE, 'r') as f:
                music_library = json.load(f)
            logger.info(f"Loaded library with {len(music_library)} tracks")
    except Exception as e:
        logger.error(f"Error loading library: {e}")


def extract_audio_features(file_path: str) -> AudioFeatures:
    """Extract audio features using AudioProcessor."""
    features = audio_processor.process_audio(file_path)
    return features


def extract_and_index_embedding(file_path: str, track_id: str, metadata: dict):
    """Extract embedding and add to FAISS index."""
    try:
        embedding = embedding_pipeline.extract_embedding(file_path)
        
        metadata_with_embedding = {
            'track_id': track_id,
            'title': metadata['title'],
            'artist': metadata['artist'],
            'bpm': metadata['bpm'],
            'key': metadata['key'],
            'rms_energy': metadata['rms_energy'],
            'filename': metadata['filename']
        }
        
        embedding_pipeline.add_to_index(embedding, metadata_with_embedding, track_id)
        logger.info(f"Indexed embedding for {track_id}")
    except Exception as e:
        logger.error(f"Error indexing embedding: {e}")


# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("DJ Recommender API starting up...")
    load_library()
    
    # Try to load existing FAISS index
    try:
        embedding_pipeline.load_index(FAISS_INDEX_PATH)
        logger.info("Loaded existing FAISS index")
    except:
        logger.info("Creating new FAISS index")
        embedding_pipeline.create_faiss_index(512)  # OpenL3 default dimension


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "library_size": len(music_library),
        "index_stats": embedding_pipeline.get_index_stats()
    }


@app.post("/upload")
async def upload_track(
    file: UploadFile = File(...),
    title: str = None,
    artist: str = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload and analyze a track.

    Process:
    1. Save uploaded file
    2. Extract audio features (BPM, key, spectral features)
    3. Create embedding and index
    4. Return analysis results
    """
    try:
        # Validate file
        if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
            raise HTTPException(status_code=400, detail="Only audio files supported (mp3, wav, flac, ogg)")

        # Save upload
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded file: {file_path}")

        # Extract audio features
        features = extract_audio_features(file_path)

        # Create track metadata
        track_id = f"track_{len(music_library)}_{int(datetime.now().timestamp())}"
        track_metadata = {
            'id': track_id,
            'title': title or file.filename,
            'artist': artist or 'Unknown',
            'filename': file.filename,
            'file_path': file_path,
            'bpm': features.bpm,
            'key': features.key,
            'key_confidence': features.key_confidence,
            'duration': features.duration,
            'rms_energy': features.rms_energy,
            'spectral_centroid': features.spectral_centroid,
            'zero_crossing_rate': features.zero_crossing_rate,
            'uploaded_at': datetime.now().isoformat()
        }

        # Store in library
        music_library[track_id] = track_metadata

        # Add embedding extraction as background task
        background_tasks.add_task(
            extract_and_index_embedding,
            file_path,
            track_id,
            track_metadata
        )

        # Save library
        save_library()

        return {
            'status': 'success',
            'track_id': track_id,
            'analysis': {
                'bpm': features.bpm,
                'key': features.key,
                'key_confidence': features.key_confidence,
                'duration': features.duration,
                'rms_energy': features.rms_energy,
                'spectral_centroid': features.spectral_centroid,
                'zero_crossing_rate': features.zero_crossing_rate
            },
            'message': 'Audio uploaded successfully. Embedding indexing in progress...'
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tracks")
async def list_tracks():
    """List all tracks in library."""
    tracks = [
        {
            'id': tid,
            'title': metadata['title'],
            'artist': metadata['artist'],
            'bpm': metadata['bpm'],
            'key': metadata['key'],
            'duration': metadata['duration']
        }
        for tid, metadata in music_library.items()
    ]
    return {
        'count': len(tracks),
        'tracks': tracks
    }


@app.get("/track/{track_id}")
async def get_track(track_id: str):
    """Get detailed track information."""
    if track_id not in music_library:
        raise HTTPException(status_code=404, detail="Track not found")

    track = music_library[track_id]
    return {
        'id': track_id,
        'title': track['title'],
        'artist': track['artist'],
        'bpm': track['bpm'],
        'key': track['key'],
        'key_confidence': track['key_confidence'],
        'duration': track['duration'],
        'rms_energy': track['rms_energy'],
        'spectral_centroid': track['spectral_centroid'],
        'zero_crossing_rate': track['zero_crossing_rate'],
        'uploaded_at': track['uploaded_at']
    }


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """
    Get recommended transition tracks.

    Combines:
    - Harmonic compatibility
    - Tempo matching
    - Energy progression
    - AI embedding similarity
    """
    try:
        # Validate request
        if request.track_id not in music_library:
            raise HTTPException(status_code=404, detail="Current track not found")

        current_track = music_library[request.track_id]

        # Get all candidates
        candidate_tracks = [
            {**metadata, 'id': tid}
            for tid, metadata in music_library.items()
            if tid != request.track_id
        ]

        if not candidate_tracks:
            return {
                'current_track': current_track,
                'recommendations': [],
                'message': 'No candidate tracks in library',
                'generation_timestamp': datetime.now().isoformat()
            }

        # Get recommendations
        recommendations = transition_recommender.recommend_transitions(
            current_track,
            candidate_tracks,
            top_k=request.top_k,
            min_score=request.min_score
        )

        # Format recommendations
        formatted_recommendations = []
        for rec in recommendations:
            track = rec['track']
            formatted_recommendations.append({
                'track': {
                    'id': track['id'],
                    'title': track['title'],
                    'artist': track['artist'],
                    'bpm': track['bpm'],
                    'key': track['key'],
                    'duration': track['duration']
                },
                'scores': {
                    'overall': round(rec['transition_score'], 3),
                    'harmonic': round(rec['harmonic_score'], 3),
                    'tempo': round(rec['tempo_score'], 3),
                    'energy': round(rec['energy_score'], 3),
                    'embedding_similarity': round(rec['embedding_similarity'], 3)
                },
                'explanation': rec['explanation'],
                'mix_strategy': transition_recommender.get_mix_strategy(
                    current_track,
                    track,
                    TransitionScore(
                        rec['harmonic_score'],
                        rec['energy_score'],
                        rec['tempo_score'],
                        rec['embedding_similarity'],
                        rec['transition_score'],
                        rec['explanation']
                    )
                )
            })

        return {
            'current_track': {
                'id': current_track['id'],
                'title': current_track['title'],
                'artist': current_track['artist'],
                'bpm': current_track['bpm'],
                'key': current_track['key']
            },
            'recommendations': formatted_recommendations,
            'generation_timestamp': datetime.now().isoformat()
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/harmonic-compatible/{key}")
async def get_harmonic_keys(key: str):
    """Get compatible keys for harmonic mixing."""
    try:
        from transition_recommender import HarmonicMixer
        compatible = HarmonicMixer.get_compatible_keys(key, num_suggestions=6)
        return {
            'input_key': key,
            'compatible_keys': [
                {'key': k, 'score': float(s)} for k, s in compatible
            ]
        }
    except Exception as e:
        logger.error(f"Error getting harmonic keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/save-index")
async def save_index():
    """Manually save FAISS index to disk."""
    try:
        embedding_pipeline.save_index(FAISS_INDEX_PATH)
        return {
            'status': 'success',
            'message': f'Index saved to {FAISS_INDEX_PATH}',
            'index_stats': embedding_pipeline.get_index_stats()
        }
    except Exception as e:
        logger.error(f"Error saving index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/track/{track_id}")
async def delete_track(track_id: str):
    """Delete a track from library."""
    try:
        if track_id not in music_library:
            raise HTTPException(status_code=404, detail="Track not found")

        track = music_library[track_id]
        
        # Delete file
        if os.path.exists(track['file_path']):
            os.remove(track['file_path'])
            logger.info(f"Deleted file: {track['file_path']}")

        # Remove from library
        del music_library[track_id]
        save_library()

        return {
            'status': 'success',
            'message': f"Deleted track: {track['title']}"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting track: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        'total_tracks': len(music_library),
        'index_stats': embedding_pipeline.get_index_stats(),
        'audio_processor_config': {
            'sample_rate': audio_processor.sr,
            'hop_length': audio_processor.hop_length
        },
        'embedding_model': 'openl3-mel256',
        'library_storage_path': LIBRARY_STORAGE,
        'upload_directory': UPLOAD_DIR
    }


# Mount frontend static files if they exist
frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
