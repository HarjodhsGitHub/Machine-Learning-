"""
Data preprocessing and library management utilities.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LibraryManager:
    """Manage music library persistence and operations."""

    def __init__(self, storage_path: str = './data/library.json'):
        self.storage_path = storage_path
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        self.library = self._load()

    def _load(self) -> Dict:
        """Load library from JSON."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    library = json.load(f)
                logger.info(f"Loaded library with {len(library)} tracks")
                return library
            except Exception as e:
                logger.error(f"Error loading library: {e}")
                return {}
        return {}

    def save(self) -> None:
        """Save library to JSON."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.library, f, indent=2)
            logger.info("Library saved")
        except Exception as e:
            logger.error(f"Error saving library: {e}")

    def add_track(self, track_id: str, metadata: Dict) -> None:
        """Add track to library."""
        self.library[track_id] = metadata
        self.save()

    def get_track(self, track_id: str) -> Optional[Dict]:
        """Get track metadata."""
        return self.library.get(track_id)

    def get_all_tracks(self) -> List[Dict]:
        """Get all tracks."""
        return list(self.library.values())

    def delete_track(self, track_id: str) -> bool:
        """Delete track from library."""
        if track_id in self.library:
            del self.library[track_id]
            self.save()
            return True
        return False

    def get_by_key(self, key: str) -> List[Dict]:
        """Get all tracks in a key."""
        return [t for t in self.library.values() if t.get('key') == key]

    def get_by_bpm_range(self, min_bpm: float, max_bpm: float) -> List[Dict]:
        """Get tracks within BPM range."""
        return [
            t for t in self.library.values()
            if min_bpm <= t.get('bpm', 0) <= max_bpm
        ]

    def export_playlist(self, output_path: str, track_ids: List[str]) -> None:
        """Export playlist as M3U format."""
        try:
            with open(output_path, 'w') as f:
                f.write("#EXTM3U\n")
                for track_id in track_ids:
                    track = self.get_track(track_id)
                    if track:
                        duration = int(track.get('duration', 0))
                        path = track.get('file_path', '')
                        title = f"{track['artist']} - {track['title']}"
                        f.write(f"#EXTINF:{duration},{title}\n")
                        f.write(f"{path}\n")
            logger.info(f"Playlist exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting playlist: {e}")

    def get_statistics(self) -> Dict:
        """Get library statistics."""
        tracks = self.library.values()
        if not tracks:
            return {'total_tracks': 0}

        bpms = [t.get('bpm', 0) for t in tracks]
        durations = [t.get('duration', 0) for t in tracks]
        
        return {
            'total_tracks': len(tracks),
            'avg_bpm': sum(bpms) / len(bpms),
            'min_bpm': min(bpms),
            'max_bpm': max(bpms),
            'total_duration_seconds': sum(durations),
            'total_duration_hours': sum(durations) / 3600
        }
