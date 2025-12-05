import requests

import re

import os

import time

import subprocess

import json

from urllib.parse import quote_plus, urljoin

from requests.adapters import HTTPAdapter

from urllib3.util.retry import Retry

from typing import List, Dict, Optional, Tuple

import importlib

import sys


# YouTube search and download configuration

YOUTUBE_SEARCH_BASE = "https://www.youtube.com/results"


# Keywords that indicate audio/lyric versions (prioritized)

PREFERRED_KEYWORDS = [

    "lyrics", "audio", "official audio", 

]


# Keywords to avoid (music videos, live performances, etc.)

AVOID_KEYWORDS = [

    "music video", "mv", "live", "concert", 

    "performance", "tour", "choreography", "dance", "remix"

]




tracks_list = [

    "fred again.. - Beto's Horns"

]


def get_video_info(youtube_url: str) -> Optional[Dict[str, str]]:

    """Return video metadata (title, uploader) using yt-dlp --dump-json."""

    try:

        cmd = ['yt-dlp', '--dump-json', youtube_url]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)

        if proc.returncode == 0 and proc.stdout:

            try:

                info = json.loads(proc.stdout)

                return {

                    'title': info.get('title'),

                    'uploader': info.get('uploader') or info.get('channel')

                }

            except Exception as e:

                print(f"Warning: failed to parse yt-dlp json: {e}")

                return None

    except Exception as e:

        print(f"Warning: failed to run yt-dlp metadata fetch: {e}")

    return None




def extract_artist_song_from_title(title: str, uploader: Optional[str] = None) -> Tuple[str, str]:

    """

    Try to extract (artist, song) from a YouTube title string.

    Falls back to (uploader, title) when parsing fails.

    """

    if not title:

        return (uploader or "Unknown Artist", "")


    # Normalize separators

    separators = [' - ', ' – ', ' — ', ':', '|']

    for sep in separators:

        if sep in title:

            parts = [p.strip() for p in title.split(sep, 1)]

            if len(parts) == 2 and parts[0] and parts[1]:

                artist = parts[0]

                song = parts[1]

                # Remove common suffixes

                song = re.sub(r"\s*\(.*?(official|audio|lyrics|lyric).*?\)", '', song, flags=re.IGNORECASE)

                song = re.sub(r"\s*\[.*?(official|audio|lyrics|lyric).*?\]", '', song, flags=re.IGNORECASE)

                song = song.strip()

                return (artist, song)


    # If no separator found, try patterns like 'Artist Song' heuristics using uploader

    if uploader and uploader.lower() in title.lower():

        # remove uploader from title

        song = re.sub(re.escape(uploader), '', title, flags=re.IGNORECASE).strip(' -:|')

        return (uploader, song if song else title)


    # Last resort: treat full title as song and uploader as artist

    return (uploader or "Unknown Artist", title.strip())

def search_youtube_with_ytdlp(query: str, max_results: int = 10) -> List[Dict[str, str]]:

    """

    Search YouTube using yt-dlp directly (more reliable than HTML scraping)

    """

    try:

        cmd = [

            'yt-dlp',

            '--get-title',

            '--get-id',

            '--flat-playlist',

            '--no-warnings',

            f'ytsearch{max_results}:{query}'

        ]

        

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        

        if result.returncode == 0:

            lines = result.stdout.strip().split('\n')

            results = []

            

            # yt-dlp returns alternating lines: title, id, title, id, ...

            for i in range(0, len(lines), 2):

                if i + 1 < len(lines):

                    title = lines[i]

                    video_id = lines[i + 1]

                    results.append({

                        'video_id': video_id,

                        'title': title,

                        'url': f"https://www.youtube.com/watch?v={video_id}",

                        'search_query': query

                    })

            

            return results

        else:

            print(f"yt-dlp search failed: {result.stderr}")

            return []

            

    except Exception as e:

        print(f"yt-dlp search failed: {e}")

        return []


def search_youtube_for_song(song_name: str, artist: str = "", max_results: int = 10) -> List[Dict[str, str]]:

    """

    Search YouTube for a song and return a list of video results.

    Prioritizes lyrics versions over music videos.

    """

    # Construct search query

    if artist:

        query = f"{artist} {song_name}"

    else:

        query = song_name


    # First try yt-dlp based search (more robust vs. YouTube HTML changes / bot blocks)

    try:

        ytdlp_results = search_youtube_with_ytdlp(query, max_results=max_results)

        if ytdlp_results:

            scored = score_youtube_results_simple(ytdlp_results, song_name, artist)

            return scored[:max_results]

    except Exception as e:

        print(f"yt-dlp search failed (falling back to scraping): {e}")


    # Search with lyrics preference

    search_queries = [

        f"{query} lyrics",

        f"{query} audio",

        query  # fallback to basic search

    ]


    results = []

    session = create_session_with_retry()


    for search_query in search_queries:

        if len(results) >= max_results:

            break


        print(f"Searching YouTube for: '{search_query}'")


        try:

            # Encode the search query for URL

            encoded_query = quote_plus(search_query)

            search_url = f"{YOUTUBE_SEARCH_BASE}?search_query={encoded_query}"


            # Get the search results page

            response = session.get(search_url, timeout=(10, 30))


            # Debug: special handling for 403 responses

            if response.status_code == 403:

                print("ERROR: Received 403 Forbidden from YouTube search.")

                try:

                    print("Request headers sent:", dict(response.request.headers))

                except Exception:

                    pass

                try:

                    print("Response headers:", dict(response.headers))

                except Exception:

                    pass

                print("Response snippet:", response.text[:1000])

                ua = response.request.headers.get("User-Agent", "<none>") if getattr(response, "request", None) else "<none>"

                print(f"To reproduce with curl:\n  curl -I -A \"{ua}\" \"{search_url}\"")

                # continue to next query / fallback

                continue


            response.raise_for_status()


            # Parse video information from the page

            video_results = parse_youtube_search_results(response.text, search_query)


            # Score and sort results based on preferences

            scored_results = score_youtube_results_simple(video_results, song_name, artist)


            # Add new results (avoid duplicates)

            existing_ids = {r.get('video_id') for r in results}

            for result in scored_results:

                if result['video_id'] not in existing_ids and len(results) < max_results:

                    results.append(result)


            # If we found good results with lyrics keywords, prefer those

            if "lyrics" in search_query.lower() and results:

                break


        except Exception as e:

            print(f"Error searching for '{search_query}': {e}")

            continue


    return results[:max_results]






def parse_youtube_search_results(html_content: str, search_query: str) -> List[Dict[str, str]]:

    """

    Parse YouTube search results from HTML content.

    """

    results = []

    

    # Pattern to match video data in YouTube's JSON

    video_pattern = r'"videoId":"([A-Za-z0-9_-]{11})".*?"title":{"runs":\[{"text":"([^"]+)"}'

    matches = re.findall(video_pattern, html_content)

    

    # Alternative pattern for different YouTube layouts

    if not matches:

        video_pattern2 = r'/watch\?v=([A-Za-z0-9_-]{11})[^"]*"[^>]*>([^<]+)</a>'

        matches = re.findall(video_pattern2, html_content)

    

    for match in matches:

        video_id, title = match

        if video_id and title:

            results.append({

                'video_id': video_id,

                'title': title.strip(),

                'url': f"https://www.youtube.com/watch?v={video_id}",

                'search_query': search_query

            })

    

    return results






def score_youtube_results_simple(results: List[Dict[str, str]], song_name: str, artist: str = "") -> List[Dict[str, str]]:

    """

    Simple scoring for fallback results when no official audio is found.

    """

    def calculate_simple_score(result):

        title = result['title'].lower()

        score = 0

        

        # Bonus for preferred keywords

        if "lyrics" in title:

            score += 20

        if "audio" in title:

            score += 15

        if "official" in title:

            score += 10

        

        # Penalty for avoided keywords

        for keyword in AVOID_KEYWORDS:

            if keyword in title:

                score -= 10

        

        # Bonus if song name is in title

        if song_name.lower() in title:

            score += 10

        

        # Bonus if artist is in title

        if artist and artist.lower() in title:

            score += 10

        

        result['score'] = score

        return score

    

    # Score and sort

    for result in results:

        calculate_simple_score(result)

    

    return sorted(results, key=lambda x: x.get('score', 0), reverse=True)


def extract_youtube_id(url: str) -> str:

    """

    Extracts the YouTube video ID from a URL.

    Supports URLs like:

    - https://www.youtube.com/watch?v=VIDEOID

    - https://youtu.be/VIDEOID

    """

    patterns = [

        r"(?:v=)([A-Za-z0-9_-]{11})",

        r"youtu\.be\/([A-Za-z0-9_-]{11})"

    ]

    for pat in patterns:

        m = re.search(pat, url)

        if m:

            return m.group(1)

    raise ValueError("Could not parse YouTube video ID from URL")


def create_session_with_retry():

    """

    Creates a requests session with retry strategy and browser-like headers to avoid 403.

    """

    session = requests.Session()

    retry_strategy = Retry(

        total=3,

        backoff_factor=1,

        status_forcelist=[429, 500, 502, 503, 504],

    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("http://", adapter)

    session.mount("https://", adapter)


    # Browser-like headers to reduce bot detection

    session.headers.update({

        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "

                       "AppleWebKit/537.36 (KHTML, like Gecko) "

                       "Chrome/120.0.0.0 Safari/537.36"),

        "Accept-Language": "en-US,en;q=0.9",

        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",

        "Referer": "https://www.youtube.com/"

    })


    # Helpful cookie to reduce consent blocks

    session.cookies.update({"CONSENT": "YES+1"})


    return session


def convert_to_mp3(video_id: str) -> str:

    """

    Calls the API to convert to mp3.

    Returns the download URL (or raises on failure).

    """

    # External API-based conversion has been deprecated/disabled in this

    # codebase because the remote service was unreliable and often unreachable.

    # If you need a conversion URL, use the local yt-dlp flow instead. This

    # stub prevents accidental usage of an undefined API_BASE variable.

    raise RuntimeError(

        "convert_to_mp3 is disabled: the external converter API is not used. "

        "Use the yt-dlp based download flow instead (get_video_info / download_with_ytdlp)."

    )


def download_file(url: str, dest_path: str):

    """

    Downloads from a URL (streaming) to a local file.

    """

    print(f"Downloading from: {url}")

    print(f"Saving to: {dest_path}")

    

    session = create_session_with_retry()

    

    try:

        with session.get(url, stream=True, timeout=(10, 300)) as r:  # 5min read timeout for large files

            r.raise_for_status()

            

            # Get file size if available

            total_size = int(r.headers.get('content-length', 0))

            if total_size > 0:

                print(f"File size: {total_size / (1024*1024):.2f} MB")

            

            downloaded = 0

            with open(dest_path, "wb") as f:

                for chunk in r.iter_content(chunk_size=8192):

                    if chunk:

                        f.write(chunk)

                        downloaded += len(chunk)

                        

                        # Simple progress indicator

                        if total_size > 0:

                            progress = (downloaded / total_size) * 100

                            print(f"\rProgress: {progress:.1f}%", end="", flush=True)

            

            print(f"\nDownload completed: {downloaded / (1024*1024):.2f} MB")

            

    except requests.exceptions.RequestException as e:

        print(f"Download failed: {e}")

        # Clean up partial file

        if os.path.exists(dest_path):

            os.remove(dest_path)

        raise RuntimeError(f"Failed to download file: {e}")


def download_with_ytdlp(youtube_url: str, output_path: str = "./", artist_song_name: str = None) -> bool:

    """

    Download YouTube video as MP3 using yt-dlp with proper "Artist - Song" naming.

    """

    try:

        # Check if yt-dlp is installed

        result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)

        if result.returncode != 0:

            print("yt-dlp is not installed. Install it with: pip install yt-dlp")

            return False


        # Ensure output path exists

        os.makedirs(output_path, exist_ok=True)


        # Try to get video metadata first so we can name the file using YouTube title

        video_info = None

        try:

            video_info = get_video_info(youtube_url)

        except Exception:

            video_info = None


        # If we have metadata, extract artist and song from the YouTube title

        final_name = None

        if video_info and video_info.get('title'):

            artist, song = extract_artist_song_from_title(video_info.get('title', ''), video_info.get('uploader'))

            candidate = f"{artist} - {song}" if artist and song else None

            if candidate:

                final_name = candidate


        # If caller provided a name, prefer the YouTube-derived name but fall back

        if artist_song_name and not final_name:

            final_name = artist_song_name


        if final_name:

            # Clean up the filename (remove special characters but keep " - ")

            clean_filename = re.sub(r"[^\w\s\-()]", '', final_name)

            clean_filename = re.sub(r"\s+", ' ', clean_filename).strip()

            output_template = os.path.join(output_path, f"{clean_filename}.%(ext)s")

        else:

            # Generic fallback template: uploader - title

            output_template = os.path.join(output_path, "%(uploader)s - %(title)s.%(ext)s")


        cmd = [

            'yt-dlp',

            '-x',  # Extract audio

            '--audio-format', 'mp3',

            '--audio-quality', '0',  # Best quality

            '--output', output_template,

            '--no-playlist',  # Only download single video

            '--restrict-filenames',  # Use safer characters in filenames

            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',

            '--referer', 'https://www.youtube.com/',

            youtube_url

        ]


        print(f"Downloading: {youtube_url}")

        result = subprocess.run(cmd, capture_output=True, text=True)


        if result.returncode == 0:

            print("✓ Download successful!")

            return True

        else:

            print("✗ Download failed!")

            print("Error:", result.stderr)

            return False

            

    except Exception as e:

        print(f"Error using yt-dlp: {e}")

        return False


def batch_download_songs(song_list: List[str], output_path: str = "./downloads") -> Dict[str, str]:

    """

    Download multiple songs by searching for them on YouTube.

    song_list can be in format: ["Song Name", "Artist - Song Name", ...]

    """

    if not os.path.exists(output_path):

        os.makedirs(output_path)

    

    results = {}

    

    for i, song_entry in enumerate(song_list, 1):

        print(f"\n{'='*50}")

        print(f"Processing {i}/{len(song_list)}: {song_entry}")

        print('='*50)

        

        try:

            # Parse song entry

            if " - " in song_entry:

                artist, song_name = song_entry.split(" - ", 1)

            else:

                artist = ""

                song_name = song_entry

            

            # Search YouTube

            search_results = search_youtube_for_song(song_name, artist, max_results=5)

            

            if not search_results:

                print(f"✗ No YouTube results found for: {song_entry}")

                results[song_entry] = "No results found"

                continue

            

            # Display options to user (or auto-select best match)

            print(f"\nFound {len(search_results)} results:")

            for j, result in enumerate(search_results):

                score_indicator = "★" * min(5, max(1, result.get('score', 0) // 2))

                

                # Highlight lyrics and best match

                title_display = result['title']

                if "lyrics" in result['title'].lower():

                    title_display = f"🎵 {title_display}"

                if j == 0:  # Best match

                    title_display = f"👑 {title_display}"

                

                print(f"  {j+1}. {title_display} {score_indicator}")

            

            # Auto-select the best result (highest score)

            best_result = search_results[0]

            print(f"\n→ Auto-selecting best match: {best_result['title']}")

            

            # Download the selected video with proper "Artist - Song" naming

            if " - " in song_entry:

                # Already in "Artist - Song" format

                formatted_name = song_entry

            else:

                # Try to format as "Unknown Artist - Song"

                formatted_name = f"Unknown Artist - {song_entry}"

            

            success = download_with_ytdlp(

                best_result['url'], 

                output_path, 

                artist_song_name=formatted_name

            )

            

            if success:

                results[song_entry] = f"Downloaded: {best_result['title']}"

            else:

                results[song_entry] = f"Failed to download: {best_result['title']}"

                

        except Exception as e:

            print(f"✗ Error processing {song_entry}: {e}")

            results[song_entry] = f"Error: {e}"

        

        # Small delay to be respectful to YouTube

        time.sleep(1)

    

    return results


def interactive_song_search():

    """

    Interactive mode for searching and downloading individual songs.

    """

    while True:

        print("\n" + "="*50)

        print("🎵 Interactive Song Search & Download")

        print("="*50)

        

        song_query = input("Enter song name (or 'Artist - Song'): ").strip()

        if not song_query:

            break

        

        try:

            # Parse input

            if " - " in song_query:

                artist, song_name = song_query.split(" - ", 1)

            else:

                artist = ""

                song_name = song_query

            

            # Search YouTube

            print(f"\nSearching for: {song_query}")

            results = search_youtube_for_song(song_name, artist, max_results=5)

            

            if not results:

                print("❌ No results found")

                continue

            

            # Display results

            print(f"\n🎯 Found {len(results)} results:")

            for i, result in enumerate(results):

                score = result.get('score', 0)

                score_stars = "★" * min(5, max(1, score // 2)) if score > 0 else "☆"

                

                # Highlight lyrics versions

                title_display = result['title']

                if "lyrics" in result['title'].lower():

                    title_display = f"🎵 {title_display}"

                

                print(f"  {i+1}. {title_display} {score_stars}")

            

            # Let user choose

            try:

                choice = input(f"\nSelect (1-{len(results)}) or Enter for best match: ").strip()

                if choice == "":

                    selected = results[0]

                else:

                    selected = results[int(choice) - 1]

            except (ValueError, IndexError):

                print("Invalid choice, using best match")

                selected = results[0]

            

            print(f"\n📥 Downloading: {selected['title']}")

            

            # Format the filename as "Artist - Song"

            if " - " in song_query:

                formatted_name = song_query

            else:

                formatted_name = f"Unknown Artist - {song_query}"

            

            # Download

            success = download_with_ytdlp(selected['url'], "./downloads", artist_song_name=formatted_name)

            

            if success:

                print("✅ Download completed!")

            else:

                print("❌ Download failed!")

                

        except Exception as e:

            print(f"❌ Error: {e}")

        

        # Ask if user wants to continue

        if input("\nSearch for another song? (y/N): ").lower() != 'y':

            break


def batch_mode():

    """

    Batch mode for downloading multiple songs from a list.

    """

    print("\n" + "="*50)

    print("📚 Batch Download Mode")

    print("="*50)

    print("Enter songs one per line. Formats supported:")

    print("  - 'Song Name'")

    print("  - 'Artist - Song Name'")

    print("Enter empty line when done.\n")

    

    songs = []

    while True:

        song = input(f"Song {len(songs)+1}: ").strip()

        if not song:

            break

        songs.append(song)

    

    if not songs:

        print("No songs entered")

        return

    

    print(f"\n📋 Processing {len(songs)} songs...")

    

    # Create downloads directory

    output_dir = "./downloads"

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

    

    # Process all songs

    results = batch_download_songs(songs, output_dir)

    

    # Print summary

    print(f"\n{'='*50}")

    print("📊 DOWNLOAD SUMMARY")

    print('='*50)

    

    successful = 0

    for song, status in results.items():

        if "Downloaded:" in status:

            print(f"✅ {song}")

            successful += 1

        else:

            print(f"❌ {song} - {status}")

    

    print(f"\n📈 Success rate: {successful}/{len(songs)} ({successful/len(songs)*100:.1f}%)")

    print(f"📁 Files saved to: {os.path.abspath(output_dir)}")


def download_dj_starter_pack():

    """

    Download the predefined DJ starter tracks collection.

    """

    print("\n" + "="*60)

    print("🎧 MUSIC LIST DOWNLOAD")

    print("="*60)

    print(f"📦 Ready to download {len(tracks_list)} tracks")

    

    confirm = input(f"\nDownload all {len(tracks_list)} tracks? (y/N): ").lower()

    if confirm != 'y':

        print("❌ Download cancelled")

        return

    

    print(f"\n🚀 Starting list download...")

    print("="*60)

    

    # Create specialized directory for DJ tracks

    output_dir = "./song_download"

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

    

    # Download all tracks

    results = batch_download_songs(tracks_list, output_dir)

    

    # Enhanced summary for DJ pack

    print(f"\n{'='*60}")

    print("🎧 MUSIC LIST DOWNLOAD SUMMARY")

    print('='*60)

    

    successful = 0

    failed_tracks = []

    

    for song, status in results.items():

        if "Downloaded:" in status:

            successful += 1

        else:

            failed_tracks.append(song)

    

    success_rate = (successful / len(tracks_list)) * 100

    print(f"📊 Downloaded: {successful}/{len(tracks_list)} tracks ({success_rate:.1f}%)")

    print(f"📁 Location: {os.path.abspath(output_dir)}")

    

    if failed_tracks:

        print(f"\n❌ Failed downloads ({len(failed_tracks)} tracks):")

        for track in failed_tracks[:10]:  # Show first 10 failed

            print(f"   • {track}")

        if len(failed_tracks) > 10:

            print(f"   • ... and {len(failed_tracks) - 10} more")

        

        retry = input(f"\nRetry failed downloads? (y/N): ").lower()

        if retry == 'y':

            print(f"\n🔄 Retrying {len(failed_tracks)} failed tracks...")

            retry_results = batch_download_songs(failed_tracks, output_dir)

            

            retry_success = sum(1 for status in retry_results.values() if "Downloaded:" in status)

            print(f"✅ Retry successful for {retry_success}/{len(failed_tracks)} tracks")


    print(f"\n🎉 Music list ready!")

    print("💡 Pro tip: Organize these by BPM and key for better mixing!")


def url_mode():

    """

    Original URL mode for downloading from direct YouTube URLs.

    """

    print("\n" + "="*50)

    print("🔗 Direct URL Download Mode")

    print("="*50)

    

    youtube_url = input("Enter YouTube URL: ").strip()

    

    if not youtube_url:

        print("❌ No URL provided")

        return

    

    print(f"📥 Downloading from: {youtube_url}")

    

    # For direct URL downloads, let yt-dlp extract artist and title

    success = download_with_ytdlp(youtube_url, "./downloads")

    

    if success:

        print("✅ Download completed!")

    else:

        print("❌ Download failed!")


def main():

    print("🎵 YouTube to MP3 Downloader with Smart Search")

    print("=" * 60)

    

    while True:

        print("\nSelect mode:")

        print("1. 🔍 Interactive Search (search by song name)")

        print("2. 📚 Batch Download (multiple songs)")

        print("3. 🎧 Music list (download all listed tracks)")

        print("4. 🔗 Direct URL Download")

        print("5. ❌ Exit")

        

        try:

            choice = input("\nEnter choice (1-5): ").strip()

            

            if choice == "1":

                interactive_song_search()

            elif choice == "2":

                batch_mode()

            elif choice == "3":

                download_dj_starter_pack()

            elif choice == "4":

                url_mode()

            elif choice == "5":

                print("👋 Goodbye!")

                break

            else:

                print("❌ Invalid choice, please try again")

                

        except KeyboardInterrupt:

            print("\n\n👋 Operation cancelled by user. Goodbye!")

            break

        except Exception as e:

            print(f"❌ Unexpected error: {e}")

            print("Please try again")


if __name__ == "__main__":

    main()