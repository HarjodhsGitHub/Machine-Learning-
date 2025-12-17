# Machine Learning + Music Tools Monorepo

AI-powered music tooling, file utilities, and ML notebooks collected in a single workspace. The highlight is a DJ Transition Recommender web app that analyzes audio and suggests harmonic, tempo-aware transitions. Utilities also include an image→PDF converter and simple audio download/convert scripts. An Iris notebook demonstrates classic ML workflows.

## What This Project Does

- DJ Transition Recommender: FastAPI app that extracts BPM, key, energy, and OpenL3 embeddings to recommend smooth track transitions with explainable scores and an included single-page UI.
- File Handler: Lightweight CLI to convert images to PDFs, including merge to a single PDF.
- Music Utilities: Scripts to batch-convert `.webm` to `.mp3` and help with audio downloads.
- Iris Demo: A Jupyter notebook showcasing an ML workflow on the Iris dataset.

## Why It’s Useful

- End-to-end example of ML for audio: from feature extraction to recommendation, with a usable UI.
- Practical utilities that speed up day-to-day workflows (image→PDF conversion, audio format conversion).
- Clear, modular structure so you can use only the parts you need.

## Repository Layout

```
File_Handler/
  requirements.txt
  pre_conv/
    converter.py
    README.md
  post_conv/
Iris/
  iris2.ipynb
Music/
  converter.py            # .webm → .mp3 batch conversion (moviepy/ffmpeg)
  downloaded.py           # YouTube search + download helper (yt-dlp)
  Transition_reccomender/
    dj_recommender/       # Full web app (FastAPI + frontend)
      backend/
      frontend/
      README.md
      INSTALLATION.md
      ARCHITECTURE.md
      PROJECT_SUMMARY.md
      requirements.txt
      run.bat / run.sh
```

## Getting Started

### Prerequisites

- Python 3.10+
- ffmpeg on PATH (required for audio tasks like conversion)
- Optional: Redis (only if you plan to run Celery workers for background tasks)

### 1) DJ Transition Recommender (Web App)

Quickstart on Windows (PowerShell):

```powershell
cd Music/Transition_reccomender/dj_recommender
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
./run.bat
```

Then open http://localhost:8000 in your browser. See detailed docs in [Music/Transition_reccomender/dj_recommender/README.md](Music/Transition_reccomender/dj_recommender/README.md). API docs are available at `/docs` when running.

Key files and guides:
- Architecture: [Music/Transition_reccomender/dj_recommender/ARCHITECTURE.md](Music/Transition_reccomender/dj_recommender/ARCHITECTURE.md)
- Installation: [Music/Transition_reccomender/dj_recommender/INSTALLATION.md](Music/Transition_reccomender/dj_recommender/INSTALLATION.md)
- Project Summary: [Music/Transition_reccomender/dj_recommender/PROJECT_SUMMARY.md](Music/Transition_reccomender/dj_recommender/PROJECT_SUMMARY.md)

### 2) File Handler (image → PDF)

Install dependency:

```bash
pip install -r File_Handler/requirements.txt
```

Common usage:

```bash
# Convert all images in default pre_conv dir → PDFs in post_conv
python File_Handler/pre_conv/converter.py --input-type image --output-type pdf

# Convert images in a specific folder to individual PDFs
python File_Handler/pre_conv/converter.py File_Handler/pre_conv/sample_images \
  --input-type image --output-type pdf

# Merge multiple images into a single PDF
python File_Handler/pre_conv/converter.py File_Handler/pre_conv/sample_images \
  --input-type image --output-type pdf --merge

# Convert a single image to PDF into post_conv
python File_Handler/pre_conv/converter.py File_Handler/pre_conv/sample_images/example.jpg \
  --output File_Handler/post_conv
```

Details: [File_Handler/pre_conv/README.md](File_Handler/pre_conv/README.md)

### 3) Music Utilities

Convert `.webm` files to `.mp3` (requires `moviepy` and `ffmpeg`):

```bash
pip install moviepy
python Music/converter.py
```

By default this converts files from `Music/pre_conversion` to `Music/converted_mp3`. Adjust paths in the script or pass your own when calling `convert_webm_to_mp3()`.

Downloader helper (uses `yt-dlp`):

```bash
pip install yt-dlp
# Edit tracks in Music/downloaded.py (see `tracks_list`) and run
python Music/downloaded.py
```

### 4) Iris Notebook

Open and run the notebook with Jupyter:

```bash
pip install jupyter
jupyter lab
# Then open Iris/iris2.ipynb
```

## Where To Get Help

- DJ Recommender docs: [Music/Transition_reccomender/dj_recommender/README.md](Music/Transition_reccomender/dj_recommender/README.md)
- Architecture details: [Music/Transition_reccomender/dj_recommender/ARCHITECTURE.md](Music/Transition_reccomender/dj_recommender/ARCHITECTURE.md)
- Installation notes: [Music/Transition_reccomender/dj_recommender/INSTALLATION.md](Music/Transition_reccomender/dj_recommender/INSTALLATION.md)
- Project summary: [Music/Transition_reccomender/dj_recommender/PROJECT_SUMMARY.md](Music/Transition_reccomender/dj_recommender/PROJECT_SUMMARY.md)

If something doesn’t work:
- Verify Python and ffmpeg are installed and on PATH
- For the web app, check the interactive API docs at `http://localhost:8000/docs`
- Open an issue in this repository with logs and reproduction steps

## Maintainers & Contributions

- Maintainer: HarjodhsGitHub
- Contributions: Issues and pull requests are welcome. Please follow the existing code style within each module and include small, focused changes. For substantial updates to the DJ Recommender, refer to the module docs first.

---

Built with FastAPI, librosa, OpenL3, FAISS, and Pillow. Enjoy! 🎵
