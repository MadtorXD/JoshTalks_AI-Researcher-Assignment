import pandas as pd
import json
import os
import re
from pathlib import Path
from scipy.io import wavfile
import concurrent.futures
from tqdm import tqdm

DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CLIPS_DIR = Path("disfluency_clips")
CLIPS_DIR.mkdir(exist_ok=True)

# 1. Load disfluencies list
print("Loading disfluencies list...")
df_disf = pd.read_csv("q2_disfluencies.csv")
disfluency_map = {} # target_word -> type

for col in df_disf.columns:
    words = df_disf[col].dropna().astype(str).tolist()
    for word in words:
        word = word.strip()
        if word:
            disfluency_map[word] = col

# Helper function to find a word safely as a whole token (or exact phrase)
def contains_disfluency(text, disfluency):
    # Escape regex specials
    escaped = re.escape(disfluency)
    # regex for whole word match + punctuation
    pattern = r"(^|[\s\.,!\?\"\'\-])(%s)([\s\.,!\?\"\'\-]|$)" % escaped
    return re.search(pattern, text) is not None

# 2. Process dataset
df_dataset = pd.read_csv("q2_dataset.csv")

def process_recording(recording_id):
    results = []
    
    transcript_path = TRANSCRIPTS_DIR / f"{recording_id}.json"
    audio_path = AUDIO_DIR / f"{recording_id}.wav"
    
    if not transcript_path.exists() or not audio_path.exists():
        return results
        
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
            
        # load audio only once if needed
        audio_loaded = False
        sr = None
        audio_data = None
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            # Find all disfluencies in this text
            found_disfluencies = []
            for word, type_name in disfluency_map.items():
                if contains_disfluency(text, word):
                    found_disfluencies.append((word, type_name))
            
            # If found, clip audio and record
            for word, type_name in found_disfluencies:
                if not audio_loaded:
                    sr, audio_data = wavfile.read(audio_path)
                    audio_loaded = True
                
                # Clip audio
                clip_filename = f"{recording_id}_seg{i}_{type_name.replace(' ', '_')}.wav".replace('"', '').replace('/', '_')
                clip_path = CLIPS_DIR / clip_filename
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Protect out of bounds
                clip_data = audio_data[start_sample:end_sample]
                if len(clip_data) > 0:
                    wavfile.write(clip_path, sr, clip_data)
                    
                    disf_formatted_type = f"{type_name} (\"{word}\")"
                    
                    results.append({
                        "recording_id": recording_id,
                        "disfluency_type": disf_formatted_type,
                        "audio_segment_url": str(clip_path),
                        "start_time (s)": start_time,
                        "end_time (s)": end_time,
                        "transcription_snippet": text,
                        "notes": ""
                    })
    except Exception as e:
        print(f"Error processing {recording_id}: {e}")
        
    return results

print("Processing recordings...")
all_results = []
recording_ids = df_dataset['recording_id'].tolist()

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_recording, rid) for rid in recording_ids]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(recording_ids)):
        all_results.extend(future.result())

# Save results
print(f"Total disfluencies detected: {len(all_results)}")
df_out = pd.DataFrame(all_results)
# Rearrange to match schema: disfluency_type,audio_segment_url,start_time (s),end_time (s),transcription_snippet,notes,recording_id
df_out = df_out[["disfluency_type", "audio_segment_url", "start_time (s)", "end_time (s)", "transcription_snippet", "notes", "recording_id"]]
df_out.to_csv("disfluencies_dataset.csv", index=False, encoding='utf-8')
print("Successfully generated disfluencies_dataset.csv and clipped audio files.")
