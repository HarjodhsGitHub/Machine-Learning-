# DJ Transition Recommender

🎵 **AI-powered DJ mixing assistant** that analyzes audio files, extracts musical features (BPM, key, energy), and recommends ideal transition tracks based on harmonic compatibility, tempo matching, and AI-learned sonic similarity.

## Features

✅ **Audio Analysis**
- BPM detection using onset detection + dynamic programming
- Musical key detection using chromagram + Krumhansl-Schmuckler algorithm
- Spectral features: MFCCs, spectral centroid, zero-crossing rate, RMS energy
- Beat grid and onset detection

✅ **AI Embeddings**
- OpenL3 deep learning embeddings for semantic audio similarity
- FAISS indexing for fast similarity search
- L2-normalized embeddings for cosine similarity

✅ **Intelligent Recommendations**
- Harmonic mixing rules (Camelot wheel compatibility)
- Tempo matching with halftime/doubletime detection
- Energy progression analysis
- Weighted scoring system combining all factors

✅ **Complete Stack**
- FastAPI backend with async processing
- Modern, responsive frontend (HTML/CSS/JS)
- Optional Celery workers for background tasks
- Persistent library storage (JSON)
- Production-ready error handling

## Project Structure

```
dj_recommender/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── audio_processor.py         # Audio analysis engine
│   ├── embedding_pipeline.py      # OpenL3 + FAISS
│   ├── transition_recommender.py  # Mixing intelligence
│   └── celery_worker.py           # Optional async tasks
├── frontend/
│   └── index.html                 # Single-page UI
├── data/
│   ├── music_index.faiss          # FAISS index (auto-created)
│   ├── music_index_metadata.pkl   # Track metadata
│   └── library.json               # Library persistence
├── uploads/                        # User-uploaded tracks
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── run.sh / run.bat               # Startup scripts
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Redis (optional, for Celery)
- ~2GB disk space (for models)

### Step 1: Clone and Setup

```bash
cd dj_recommender
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **fastapi, uvicorn** - Web framework
- **librosa, soundfile** - Audio I/O
- **openl3** - AI embeddings
- **faiss-cpu** - Similarity search
- **celery, redis** - Optional async tasks

### Step 3: Environment Configuration

```bash
cp .env.example .env
# Edit .env with your settings (optional - defaults work fine)
```

## Running the Application

### Option A: Simple Startup (Recommended for Development)

```bash
# Windows
run.bat

# macOS/Linux
bash run.sh
```

This runs:
- FastAPI backend on `http://localhost:8000`
- Frontend on `http://localhost:8000`
- Auto-opens browser

### Option B: Manual Startup

```bash
# Start backend
cd backend
python main.py

# In another terminal, open frontend
# Browser: http://localhost:8000
```

### Option C: With Celery Workers (Production)

```bash
# Terminal 1: Redis server
redis-server

# Terminal 2: FastAPI backend
cd backend
python main.py

# Terminal 3: Celery worker
cd backend
celery -A celery_worker worker --loglevel=info
```

## API Documentation

### Interactive API Docs
**Swagger UI:** http://localhost:8000/docs
**ReDoc:** http://localhost:8000/redoc

### Key Endpoints

#### Upload & Analyze Track
```bash
POST /upload
```
**Request:** Multipart form with audio file, title, artist
**Response:** Track ID, BPM, key, confidence scores

Example:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@song.mp3" \
  -F "title=Midnight Groove" \
  -F "artist=DJ Phoenix"
```

#### Get Recommendations
```bash
POST /recommend
Content-Type: application/json

{
  "track_id": "track_0_1234567890",
  "top_k": 10,
  "min_score": 0.5
}
```

**Response:**
```json
{
  "current_track": {
    "id": "track_0_1234567890",
    "title": "Midnight Groove",
    "bpm": 128.0,
    "key": "A"
  },
  "recommendations": [
    {
      "track": {
        "id": "track_1_1234567891",
        "title": "Electric Dream",
        "artist": "Sound Wizard",
        "bpm": 130.0,
        "key": "E"
      },
      "scores": {
        "overall": 0.872,
        "harmonic": 0.9,
        "tempo": 0.85,
        "energy": 0.8,
        "embedding_similarity": 0.88
      },
      "explanation": "✓ Harmonic: A → E... ✓ Tempo: 128 → 130 BPM...",
      "mix_strategy": {
        "difficulty": "Easy - Professional-level match",
        "eq": "Keys are compatible...",
        "cueing": "Cue incoming track at 0.0...",
        "fader": "Energy rising - smooth crossfader..."
      }
    }
  ],
  "generation_timestamp": "2024-01-15T10:30:00"
}
```

#### List Library
```bash
GET /tracks
```

#### Get Track Details
```bash
GET /track/{track_id}
```

#### Harmonic Compatibility
```bash
GET /harmonic-compatible/{key}
```

Example: `/harmonic-compatible/C` returns compatible keys sorted by score.

#### System Statistics
```bash
GET /stats
```

#### Save Index
```bash
GET /save-index
```

#### Delete Track
```bash
DELETE /track/{track_id}
```

## Audio Analysis Details

### BPM Detection
- Uses `librosa.beat.beat_track()` with onset detection
- Employs dynamic programming for robust estimation
- Works on majority of EDM, pop, hip-hop tracks
- Returns single float value

### Key Detection
- Computes constant-Q chroma features
- Compares against Krumhansl-Schmuckler major key profiles
- Returns key (C, C#, D, ..., B) and confidence (0-1)
- Works well for harmonically clear material

### Spectral Features
- **MFCC (13 coefficients):** Captures frequency characteristics
- **Spectral Centroid:** Average frequency (Hz)
- **Zero-Crossing Rate:** Signal noisiness
- **RMS Energy:** Loudness/dynamics indicator

### Beat Grid
- Returns frame indices of detected beats
- Useful for cueing and beatmatching
- Can be converted to timestamps using: `librosa.frames_to_time(beats, sr=22050)`

## Embedding Pipeline

### OpenL3 Model
- Pre-trained deep learning model (Cramer et al., 2019)
- Trained on large-scale audio dataset
- Generates 512-dim vectors capturing semantic similarity
- Music content-type model used

### FAISS Indexing
- Flat L2 indexing (exact search)
- Can switch to IVFFlat for large libraries (millions of tracks)
- Metadata persisted separately as pickle

### Adding Custom Tracks

Example Python script:
```python
from backend.embedding_pipeline import EmbeddingPipeline

pipeline = EmbeddingPipeline()

# Extract embedding
embedding = pipeline.extract_embedding('song.mp3')

# Add to index
metadata = {
    'title': 'My Song',
    'artist': 'Artist Name',
    'bpm': 120,
    'key': 'C'
}
pipeline.add_to_index(embedding, metadata, track_id='custom_1')

# Search
results = pipeline.search_similar(embedding, k=5)
```

## Transition Recommender Logic

### Scoring Components

1. **Harmonic Score (35% weight)**
   - Same key: 1.0
   - Half step: 0.9
   - Whole step: 0.8
   - Minor third: 0.7
   - Tritone: 0.0
   - Uses Camelot wheel positions for circular key distance

2. **Tempo Score (25% weight)**
   - Same BPM: 1.0
   - Halftime/doubletime: 1.0
   - ±5% deviation: 0.95
   - ±10% deviation: 0.85
   - ±20% deviation: 0.5
   - Penalizes extreme differences

3. **Energy Score (15% weight)**
   - Rising energy (build): High score
   - Similar energy: Medium score
   - Dropping energy (cooldown): Lower but acceptable
   - Based on RMS energy of tracks

4. **Embedding Similarity (25% weight)**
   - Cosine similarity from OpenL3 vectors
   - 0-1 range (L2 normalized)
   - Captures sonic character, instrumentation, texture

### Overall Score Calculation
```
overall_score = (
    0.35 * harmonic_score +
    0.25 * tempo_score +
    0.15 * energy_score +
    0.25 * embedding_similarity
)
```

### Mix Strategy Suggestions
- **EQ Recommendations:** Based on harmonic compatibility
- **Cueing:** Based on tempo match
- **Fader Strategy:** Based on energy progression
- **Difficulty Assessment:** Overall score interpretation

## Frontend Usage

### Uploading Tracks
1. Drag & drop audio file or click to select
2. Enter track title and artist (optional)
3. Click "Upload & Analyze"
4. Wait for analysis (takes 5-30 seconds depending on file length)

### Getting Recommendations
1. Click "Recommend" button on any track
2. System displays top 10 compatible transitions
3. Each recommendation shows:
   - Similarity score (0-100%)
   - Individual component scores (harmonic, tempo, energy, vibe)
   - Human-readable explanation
   - Mixing strategy suggestions

### Managing Library
- View all tracks with BPM, key, duration
- Delete tracks individually
- Track count and indexing status in stats

## Configuration

### Weights Adjustment
Edit `backend/main.py` line ~33:
```python
transition_recommender = TransitionRecommender(
    harmonic_weight=0.35,    # Increase for harmonic focus
    tempo_weight=0.25,
    energy_weight=0.15,
    embedding_weight=0.25    # Increase for sonic similarity focus
)
```

### Sample Rate & Hop Length
Edit `backend/audio_processor.py`:
```python
audio_processor = AudioProcessor(sr=22050, hop_length=512)
# sr: Sample rate (22050 standard for music)
# hop_length: Affects feature time resolution
```

### Recommendation Filtering
Adjust in frontend or API:
```json
{
  "track_id": "...",
  "top_k": 10,           # Number of results
  "min_score": 0.5       # Minimum quality threshold (0-1)
}
```

## Performance Tips

### For Large Libraries (1000+ tracks)

1. **Switch to Approximate FAISS:**
```python
# In embedding_pipeline.py, line ~50
index = faiss.IndexIVFFlat(d, nlist=100)  # Approximate, faster
```

2. **Enable GPU:**
```bash
pip install faiss-gpu
```

3. **Increase Celery workers:**
```bash
celery -A celery_worker worker -c 8  # 8 workers
```

### Memory Optimization
- OpenL3 model (~200MB) loaded once
- FAISS index in-memory (1.5KB per track embedding)
- Metadata loaded on startup

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
# Or individually:
pip install librosa openl3 faiss-cpu
```

### Port 8000 already in use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8000
kill -9 <PID>
```

### Audio file not uploading
- Ensure file is valid audio format (MP3, WAV, FLAC, OGG)
- Check file size (<500MB recommended)
- Check browser console (F12) for errors

### Recommendations not appearing
- Ensure at least 2 tracks uploaded
- Check if embeddings are indexed (see `/stats`)
- Increase `min_score` threshold if getting zero results

### Slow analysis
- Large files (20+ min) take longer
- OpenL3 model requires deep learning inference
- GPU acceleration available: `pip install faiss-gpu`

## Dependencies Breakdown

| Package | Purpose | Version |
|---------|---------|---------|
| fastapi | Web API | 0.104.1 |
| uvicorn | ASGI server | 0.24.0 |
| librosa | Audio processing | 0.10.0 |
| openl3 | AI embeddings | 0.4.0 |
| faiss-cpu | Similarity search | 1.7.4 |
| celery | Task queue | 5.3.4 |
| pydantic | Data validation | 2.5.0 |

## Model Details

### OpenL3
- **Input:** Raw audio (16kHz mono)
- **Output:** 512-dim embedding
- **Training:** AudioSet + FSD50K
- **License:** MIT
- **Paper:** [Look, Listen, and Learn More (Cramer et al., 2019)](https://arxiv.org/abs/1904.12294)

### Librosa
- **BPM:** Uses beat tracking with onset detection
- **Key:** Chroma-based with Krumhansl-Schmuckler profiles
- **License:** ISC
- **Paper:** [librosa: Audio and Music Signal Analysis in Python (McFee et al., 2015)](https://doi.org/10.48550/arXiv.1508.04389)

## Advanced Usage

### Batch Processing
```python
from backend.celery_worker import batch_index_task

# Index 100 tracks in background
batch_index_task.delay(
    file_paths=['track1.mp3', 'track2.mp3', ...],
    metadata_list=[{'title': 't1', 'artist': 'a1'}, ...]
)
```

### Custom Scoring Weights
```python
from backend.transition_recommender import TransitionRecommender

rec = TransitionRecommender(
    harmonic_weight=0.5,      # Prioritize harmonic matching
    embedding_weight=0.3,
    tempo_weight=0.15,
    energy_weight=0.05
)

score = rec.calculate_transition_score(track1, track2, embedding_sim=0.92)
```

### Direct Audio Processing
```python
from backend.audio_processor import AudioProcessor

processor = AudioProcessor()
features = processor.process_audio('song.mp3')

print(f"BPM: {features.bpm}")
print(f"Key: {features.key}")
print(f"Confidence: {features.key_confidence}")
```

## Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "backend/main.py"]
```

Build & run:
```bash
docker build -t dj-recommender .
docker run -p 8000:8000 -v $(pwd)/data:/app/data dj-recommender
```

### Production Server (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000
```

### With Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Contributing

Improvements welcome! Areas for enhancement:
- Genre classification integration
- BPM transition smoothing algorithms
- Waveform visualization
- Beatmatching grid visualization
- MIDI export for production
- Multi-language support

## License

MIT License - Free for personal and commercial use

## Citation

If you use this system in research or production, please cite:

```bibtex
@software{dj_recommender_2024,
  title={DJ Transition Recommender: AI-Powered Music Mixing Assistant},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dj-recommender}
}
```

## Support & Issues

- **GitHub Issues:** Report bugs and feature requests
- **Documentation:** See README.md and inline code comments
- **API Docs:** Available at `http://localhost:8000/docs`

---

**Built with ❤️ for DJs everywhere** 🎵

Made possible by:
- [librosa](https://librosa.org/) - Audio analysis
- [OpenL3](https://github.com/marl/openl3) - AI embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
