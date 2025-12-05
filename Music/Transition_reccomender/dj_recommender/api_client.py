#!/usr/bin/env python
"""
API Client for DJ Recommender - Python SDK
Use this to interact with the DJ Recommender API programmatically.
"""

import requests
import json
from typing import Dict, List, Optional
from pathlib import Path


class DJRecommenderClient:
    """Python client for DJ Recommender API."""

    def __init__(self, base_url: str = 'http://localhost:8000'):
        self.base_url = base_url
        self.session = requests.Session()

    def upload_track(
        self,
        file_path: str,
        title: Optional[str] = None,
        artist: Optional[str] = None
    ) -> Dict:
        """Upload and analyze a track."""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {}
            if title:
                data['title'] = title
            if artist:
                data['artist'] = artist
            
            response = self.session.post(
                f'{self.base_url}/upload',
                files=files,
                data=data
            )
            return response.json()

    def get_tracks(self) -> Dict:
        """Get all tracks in library."""
        response = self.session.get(f'{self.base_url}/tracks')
        return response.json()

    def get_track(self, track_id: str) -> Dict:
        """Get track details."""
        response = self.session.get(f'{self.base_url}/track/{track_id}')
        return response.json()

    def get_recommendations(
        self,
        track_id: str,
        top_k: int = 10,
        min_score: float = 0.5
    ) -> Dict:
        """Get transition recommendations."""
        payload = {
            'track_id': track_id,
            'top_k': top_k,
            'min_score': min_score
        }
        response = self.session.post(
            f'{self.base_url}/recommend',
            json=payload
        )
        return response.json()

    def get_harmonic_keys(self, key: str) -> Dict:
        """Get compatible keys."""
        response = self.session.get(
            f'{self.base_url}/harmonic-compatible/{key}'
        )
        return response.json()

    def delete_track(self, track_id: str) -> Dict:
        """Delete a track."""
        response = self.session.delete(f'{self.base_url}/track/{track_id}')
        return response.json()

    def get_stats(self) -> Dict:
        """Get system statistics."""
        response = self.session.get(f'{self.base_url}/stats')
        return response.json()

    def save_index(self) -> Dict:
        """Save FAISS index."""
        response = self.session.get(f'{self.base_url}/save-index')
        return response.json()

    def batch_upload(self, directory: str, pattern: str = '*.mp3') -> List[Dict]:
        """Upload all files matching pattern in directory."""
        results = []
        for file_path in Path(directory).glob(pattern):
            result = self.upload_track(str(file_path))
            results.append(result)
        return results


# Example usage
if __name__ == '__main__':
    client = DJRecommenderClient('http://localhost:8000')

    # Get all tracks
    print("📚 Library:")
    tracks = client.get_tracks()
    for track in tracks['tracks'][:3]:
        print(f"  - {track['title']} ({track['bpm']:.0f} BPM, {track['key']})")

    # Get recommendations for first track
    if tracks['tracks']:
        track_id = tracks['tracks'][0]['id']
        print(f"\n🎯 Recommendations for {tracks['tracks'][0]['title']}:")
        recs = client.get_recommendations(track_id, top_k=5)
        for rec in recs['recommendations'][:3]:
            print(f"  - {rec['track']['title']} ({rec['scores']['overall']*100:.0f}%)")

    # Get stats
    print(f"\n📊 Stats:")
    stats = client.get_stats()
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Indexed: {stats['index_stats']['ntotal']}")
