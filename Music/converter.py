import os
from pathlib import Path
from moviepy import AudioFileClip


def convert_webm_to_mp3(input_folder, output_folder):
    """
    Convert all WebM files in the input folder to MP3 format in the output folder.
    
    Args:
        input_folder (str): Path to the folder containing WebM files
        output_folder (str): Path to the folder where MP3 files will be saved
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all webm files from input folder
    input_path = Path(input_folder)
    webm_files = list(input_path.glob("*.webm"))
    
    if not webm_files:
        print(f"No WebM files found in {input_folder}")
        return
    
    print(f"Found {len(webm_files)} WebM file(s) to convert")
    
    # Convert each file
    for webm_file in webm_files:
        try:
            print(f"Converting: {webm_file.name}")
            
            # Load the WebM file
            audio_clip = AudioFileClip(str(webm_file))
            
            # Create output filename
            output_filename = webm_file.stem + ".mp3"
            output_path = Path(output_folder) / output_filename
            
            # Export as MP3
            audio_clip.write_audiofile(str(output_path), bitrate="192k")
            audio_clip.close()
            
            print(f"✓ Converted: {output_filename}")
            
        except Exception as e:
            print(f"✗ Error converting {webm_file.name}: {str(e)}")


if __name__ == "__main__":
    # Example usage - modify these paths as needed
    input_folder = "pre_conversion"  # Folder containing WebM files
    output_folder = "converted_mp3"  # Folder where MP3 files will be saved
    
    convert_webm_to_mp3(input_folder, output_folder)
