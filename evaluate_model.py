import torch
import evaluate
import jiwer
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
import os
import sys
import librosa
import io

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FLEURS_DATASET = "google/fleurs"
LANGUAGE = "hi_in"
WHISPER_LANG = "Hindi"

MODEL_BASELINE = "openai/whisper-small"
MODEL_TUNED = os.path.join(SCRIPT_DIR, "whisper-small-hindi")  # FIXED: Use absolute path
SAMPLE_SIZE = 200

def evaluate_model(model_name_or_path, dataset):
    print(f"\nEvaluating Model: {model_name_or_path}")
    
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=WHISPER_LANG, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    predictions = []
    references = []
    
    sample_size = min(len(dataset), SAMPLE_SIZE) 
    
    print(f"Running inference on {sample_size} audio samples...")
    
    # bypass Audio feature decoding (can be buggy with large datasets)
    table = dataset.data
    audio_col = table.column("audio")
    trans_col = table.column("transcription")
    
    for idx in tqdm(range(sample_size)):
        try:
            # Extract raw bytes from Arrow struct
            audio_struct = audio_col[idx].as_py()
            bytes_data = audio_struct["bytes"]
            array, sr = librosa.load(io.BytesIO(bytes_data), sr=16000)
            
            reference = trans_col[idx].as_py()
        except Exception as e:
            print(f"Sample {idx} error: {e}")
            continue
        
        input_features = processor(array, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, 
                forced_decoder_ids=processor.get_decoder_prompt_ids(language=WHISPER_LANG, task="transcribe")
            )
            
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        predictions.append(transcription)
        references.append(reference)
        
    wer = jiwer.wer(references, predictions)
    print(f"\nResult for {model_name_or_path}:")
    print(f"WER (Word Error Rate): {wer * 100:.2f} %")
    return wer

def main():
    print("=" * 60)
    print("FLEURS Hindi ASR Evaluation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nLoading google/fleurs dataset...")
    fleurs_hi = load_dataset(FLEURS_DATASET, LANGUAGE, split="test", trust_remote_code=True)
    print(f"FLEURS Hindi test set: {len(fleurs_hi)} samples")
    
    metric = evaluate.load("wer")
    
    # ----- Baseline -----
    print("\n" + "-" * 40)
    print("----- BASELINE EVALUATION -----")
    print("-" * 40)
    baseline_wer = evaluate_model(MODEL_BASELINE, fleurs_hi)
    
    # ----- Fine-tuned -----
    print("\n" + "-" * 40)
    print("----- FINE-TUNED MODEL EVALUATION -----")
    print("-" * 40)
    tuned_wer = None
    try:
        tuned_wer = evaluate_model(MODEL_TUNED, fleurs_hi)
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        
    # ----- Summary -----
    print("\n" + "=" * 60)
    print("FINAL WER RESULTS")
    print("=" * 60)
    print(f"{'Model':<35} {'WER':>10}")
    print("-" * 47)
    print(f"{'Baseline (whisper-small)':<35} {baseline_wer * 100:>9.2f}%")
    if tuned_wer is not None:
        print(f"{'Fine-Tuned (whisper-small-hindi)':<35} {tuned_wer * 100:>9.2f}%")
        improvement = baseline_wer - tuned_wer
        print(f"\nImprovement: {improvement * 100:.2f}% absolute WER reduction")
    print("=" * 60)

if __name__ == "__main__":
    main()
