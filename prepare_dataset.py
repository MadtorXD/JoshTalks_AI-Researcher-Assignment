import pandas as pd
import requests
import os
import json
import librosa
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# files and paths
CSV_FILE = "asr_dataset.csv"
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PROCESSED_DATASET_PATH = DATA_DIR / "processed_dataset.jsonl"

# Create directories if they do not exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# url mapping for the bucket
BASE_URL_OLD = "https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/"
BASE_URL_NEW = "https://storage.googleapis.com/upload_goai/"

def fix_url(old_url):
    """
    Transforms the old Google Cloud Storage URL format to the new format.
    Example old: https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/967179/825780_audio.wav
    Example new: https://storage.googleapis.com/upload_goai/967179/825780_audio.wav
    """
    if pd.isna(old_url):
        return None
    return str(old_url).replace(BASE_URL_OLD, BASE_URL_NEW)

def download_file(url, save_path):
    """Downloads a file from a URL to a local path."""
    if not url:
        return False
    if save_path.exists():
        return True # File already downloaded
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        # Clean up partial files
        if save_path.exists():
            save_path.unlink()
        return False

def process_row(index, row):
    """Processes a single row from the CSV dataset."""
    recording_id = row['recording_id']
    
    audio_url = fix_url(row['rec_url_gcp'])
    transcript_url = fix_url(row['transcription_url_gcp'])
    
    audio_path = AUDIO_DIR / f"{recording_id}.wav"
    transcript_path = TRANSCRIPTS_DIR / f"{recording_id}.json"
    
    # 1. Download Audio
    audio_success = download_file(audio_url, audio_path)
    
    # 2. Download Transcript JSON
    transcript_success = download_file(transcript_url, transcript_path)
    
    if not audio_success or not transcript_success:
        return None
        
    # 3. Read the downloaded transcript JSON to get the text
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
            
            text = ""
            if isinstance(transcript_data, list):
                # It's a list of utterances, concatenate them
                texts = [item.get('text', '') for item in transcript_data if 'text' in item]
                text = " ".join(texts)
            elif isinstance(transcript_data, dict):
                text = transcript_data.get('text', transcript_data.get('transcript', ''))
            
            if not text:
                print(f"Warning: No text found in {transcript_path}")
                return None
                
    except Exception as e:
         print(f"Failed to read transcript for {recording_id}: {e}")
         return None

    return {
        "recording_id": recording_id,
        "audio_path": str(audio_path.absolute()),
        "text": text,
        "duration": row['duration']
    }

def main():
    print("Loading CSV metadata...")
    df = pd.read_csv(CSV_FILE)
    print(f"Total rows in CSV: {len(df)}")
    
    processed_records = []
    
    print("Downloading and processing data (this may take a while)...")
    # Use ThreadPoolExecutor for concurrent downloads
    # The rate limits on GCS are usually generous, but we'll use 10 workers just in case.
    with ThreadPoolExecutor(max_workers=10) as executor:
        import concurrent.futures
        futures = [executor.submit(process_row, index, row) for index, row in df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df)):
            result = future.result()
            if result:
                processed_records.append(result)
                
    print(f"Successfully processed {len(processed_records)} records.")
    
    # Save the processed dataset to JSONL format for easy loading with HuggingFace datasets
    print(f"Saving dataset format to {PROCESSED_DATASET_PATH}...")
    with open(PROCESSED_DATASET_PATH, 'w', encoding='utf-8') as f:
        for record in processed_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
