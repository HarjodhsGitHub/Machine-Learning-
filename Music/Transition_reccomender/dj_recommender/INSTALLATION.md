# INSTALLATION & QUICK START GUIDE

## 🚀 5-Minute Quick Start

### Windows
```bash
cd dj_recommender
run.bat
```

### macOS / Linux
```bash
cd dj_recommender
chmod +x run.sh
bash run.sh
```

This will:
1. ✅ Create Python virtual environment
2. ✅ Install all dependencies (1-2 minutes first time)
3. ✅ Start FastAPI backend on http://localhost:8000
4. ✅ Auto-open browser with web interface

**Then:**
- Upload MP3/WAV files
- Get AI recommendations
- Mix with confidence!

---

## 📋 System Requirements

- **OS:** Windows 10+, macOS 10.13+, or Linux
- **Python:** 3.8 - 3.11 (3.9+ recommended)
- **RAM:** 4GB minimum (8GB recommended for large libraries)
- **Disk:** 3GB (1.5GB for models, 1.5GB for data)
- **Network:** Internet (first run downloads models)

---

## 🔧 Detailed Installation

### Step 1: Install Python

**Windows:**
1. Download from https://python.org
2. Check "Add Python to PATH"
3. Click "Install Now"

**macOS:**
```bash
brew install python@3.10
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Verify installation:**
```bash
python --version  # Should show 3.8+
pip --version
```

### Step 2: Extract Project

```bash
# If in zip file
unzip dj_recommender.zip
cd dj_recommender

# Or if cloned from git
cd dj_recommender
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verify (you should see `(venv)` prefix):**
```bash
python --version
```

### Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**This will install:**
- FastAPI, Uvicorn (web framework)
- Librosa, SoundFile (audio processing)
- OpenL3 (AI embeddings) - ~200MB download
- FAISS (similarity search)
- Celery, Redis (async tasks)
- Pydantic (validation)

**Installation time:** 2-5 minutes depending on internet speed

### Step 5: Create Directories

```bash
# Create necessary folders
mkdir -p uploads data

# Windows alternative:
# mkdir uploads
# mkdir data
```

### Step 6: Run Backend

```bash
cd backend
python main.py
```

**Expected output:**
```
🎵 DJ Recommender API starting up...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 7: Open Frontend

**Automatic:** Browser should open automatically
**Manual:** http://localhost:8000

---

## 🎯 First Use Tutorial

### 1. Upload Your First Track

1. Click "📤 Upload Track" section
2. Drag & drop an MP3/WAV file (or click to select)
3. Enter track title and artist
4. Click "Upload & Analyze"
5. Wait 10-30 seconds for analysis

**You'll see:**
- BPM detected
- Musical key identified
- Confidence scores
- Track added to library

### 2. Generate Recommendations

1. In library, click "Recommend" on any track
2. Wait for AI analysis (5-10 seconds)
3. See 10 recommended transitions with:
   - Overall similarity score (0-100%)
   - Individual scores (harmonic, tempo, energy, vibe)
   - Mixing tips and difficulty level

### 3. Explore & Manage

- **View all tracks:** Shows BPM, key, duration
- **Delete tracks:** Remove from library
- **Save index:** Persist embeddings for later
- **Stats:** See library size and index status

---

## 🔌 API Usage (Advanced)

### Interactive API Documentation
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### cURL Examples

**Upload track:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@song.mp3" \
  -F "title=My Track" \
  -F "artist=My Artist"
```

**Get recommendations:**
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"track_id":"track_0_1234567890","top_k":10,"min_score":0.5}'
```

**List all tracks:**
```bash
curl "http://localhost:8000/tracks"
```

### Python Client

```python
from api_client import DJRecommenderClient

client = DJRecommenderClient('http://localhost:8000')

# Upload
result = client.upload_track('song.mp3', title='My Song', artist='Artist')
print(f"BPM: {result['analysis']['bpm']}")

# Get recommendations
recs = client.get_recommendations('track_0_1234567890', top_k=5)
for rec in recs['recommendations']:
    print(f"{rec['track']['title']}: {rec['scores']['overall']*100:.0f}%")
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file (or copy `.env.example`):

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Audio Processing
UPLOAD_DIR=./uploads
SAMPLE_RATE=22050

# Embeddings
FAISS_INDEX_PATH=./data/music_index.faiss
EMBEDDING_MODEL=openl3

# Celery (optional)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
REDIS_URL=redis://localhost:6379/0
```

### Recommendation Weights

Edit `backend/main.py`:

```python
# Line ~140
transition_recommender = TransitionRecommender(
    harmonic_weight=0.35,      # For harmonic-focused mixing
    tempo_weight=0.25,         # For tempo-based matching
    energy_weight=0.15,        # For energy progression
    embedding_weight=0.25      # For sonic similarity
)
```

**Note:** Weights auto-normalize to sum = 1.0

---

## 📦 Optional: Celery Background Tasks

For async processing (useful for production):

### Install Redis

**Windows:** Download from https://github.com/microsoftarchive/redis/releases

**macOS:**
```bash
brew install redis
```

**Linux:**
```bash
sudo apt install redis-server
```

### Start Redis

**Windows:**
```bash
redis-server
```

**macOS/Linux:**
```bash
redis-server
```

### Start Celery Worker

In new terminal:
```bash
cd backend
celery -A celery_worker worker --loglevel=info
```

### Use Background Tasks

FastAPI will automatically queue embeddings to Celery.

---

## 🧪 Testing

### Run Examples
```bash
python examples.py
```

This demonstrates:
1. Audio analysis
2. Harmonic mixing rules
3. Tempo matching
4. Transition scoring
5. Embedding pipeline

### Run Unit Tests
```bash
pip install pytest
pytest test_recommender.py -v
```

---

## 🐛 Troubleshooting

### Port 8000 Already in Use

**Windows:**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**macOS/Linux:**
```bash
lsof -i :8000
kill -9 <PID>
```

Or use different port:
```bash
# Edit backend/main.py
API_PORT=8001  # In .env
```

### Module Not Found Errors

```bash
# Make sure venv is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstall requirements
pip install --force-reinstall -r requirements.txt
```

### OpenL3 Model Download Fails

The model (~200MB) downloads on first use. If connection fails:

```bash
# Manual download:
python -c "import openl3; openl3.models.load_model('mel256', 'music')"
```

### Audio File Won't Upload

- Ensure MP3/WAV/FLAC/OGG format
- Check file is valid (play in media player first)
- Max recommended: 500MB
- Check browser console for errors (F12)

### Slow Analysis

- Large files (20+ minutes) take 30-60s
- OpenL3 embedding requires deep learning (5-10s per track)
- GPU acceleration available: `pip install faiss-gpu`

### Recommendations Show Zero Results

- Need at least 2 tracks in library
- Try increasing `min_score` threshold
- Check index is populated: `/stats` endpoint

---

## 📱 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "backend/main.py"]
```

Build and run:
```bash
docker build -t dj-recommender .
docker run -p 8000:8000 -v $(pwd)/data:/app/data dj-recommender
```

### Using Nginx Reverse Proxy

```nginx
upstream dj_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://dj_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

---

## 📚 Next Steps

1. **Explore API docs:** http://localhost:8000/docs
2. **Read README.md:** Full documentation
3. **Run examples.py:** See all features in action
4. **Try API client:** Use Python SDK for automation
5. **Customize:** Adjust weights, thresholds, models

---

## 🆘 Getting Help

### Check Logs

```bash
# Terminal output shows real-time logs
# Check browser console (F12) for frontend errors
```

### View API Docs

- http://localhost:8000/docs - Interactive Swagger UI
- http://localhost:8000/redoc - Static ReDoc

### Common Issues

See **Troubleshooting** section above for:
- Port conflicts
- Module not found
- Upload failures
- Slow performance

---

## ✅ You're All Set!

You now have a complete AI-powered DJ mixing system!

🎵 **Upload tracks** → 🤖 **Get AI recommendations** → 🎧 **Mix with confidence**

**Happy mixing!** 🎚️🎛️
