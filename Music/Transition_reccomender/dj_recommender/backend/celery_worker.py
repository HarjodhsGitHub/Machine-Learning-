"""
Optional Celery worker for async audio analysis.
Offloads heavy processing from FastAPI request cycle.

To use:
1. Start Redis: redis-server
2. Start Celery worker: celery -A celery_worker worker --loglevel=info
3. FastAPI will queue tasks automatically
"""

from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Celery
celery_app = Celery(
    'dj_recommender',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minute time limit
)


@celery_app.task(bind=True, name='extract_embedding_task')
def extract_embedding_task(self, file_path: str, track_id: str, metadata: dict):
    """
    Async task to extract embedding and index.
    """
    try:
        from embedding_pipeline import EmbeddingPipeline
        
        pipeline = EmbeddingPipeline()
        
        # Load existing index
        faiss_path = os.getenv('FAISS_INDEX_PATH', './data/music_index.faiss')
        try:
            pipeline.load_index(faiss_path)
        except:
            pipeline.create_faiss_index(512)
        
        # Extract and index
        embedding = pipeline.extract_embedding(file_path)
        pipeline.add_to_index(embedding, metadata, track_id)
        
        # Save index
        pipeline.save_index(faiss_path)
        
        self.update_state(state='SUCCESS', meta={'message': f'Indexed {track_id}'})
        return {'status': 'success', 'track_id': track_id}
        
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name='process_audio_task')
def process_audio_task(self, file_path: str, track_id: str):
    """
    Async task to process audio and extract features.
    """
    try:
        from audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        features = processor.process_audio(file_path)
        
        result = {
            'track_id': track_id,
            'bpm': features.bpm,
            'key': features.key,
            'key_confidence': features.key_confidence,
            'duration': features.duration,
            'rms_energy': features.rms_energy,
            'spectral_centroid': features.spectral_centroid
        }
        
        self.update_state(state='SUCCESS', meta=result)
        return result
        
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(name='batch_index_task')
def batch_index_task(file_paths: list, metadata_list: list):
    """
    Batch indexing task for multiple tracks.
    """
    try:
        from embedding_pipeline import EmbeddingPipeline
        import numpy as np
        
        pipeline = EmbeddingPipeline()
        
        faiss_path = os.getenv('FAISS_INDEX_PATH', './data/music_index.faiss')
        try:
            pipeline.load_index(faiss_path)
        except:
            pipeline.create_faiss_index(512)
        
        embeddings = []
        for file_path in file_paths:
            embedding = pipeline.extract_embedding(file_path)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        pipeline.batch_add_embeddings(embeddings, metadata_list)
        pipeline.save_index(faiss_path)
        
        return {
            'status': 'success',
            'tracks_indexed': len(file_paths)
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
