import os
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import json
import random
from torch.utils.data import Dataset as TorchDataset
import librosa
import sys

# Log stdout to a file as well
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("training.log")
sys.stderr = sys.stdout

# Config
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "Hindi"
TASK = "transcribe"
DATASET_PATH = "data/chunked_dataset.jsonl"
OUTPUT_DIR = "whisper-small-hindi"

class WhisperHindiDataset(TorchDataset):
    def __init__(self, jsonl_path, processor, split='train', test_size=0.1, seed=42):
        self.processor = processor
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            all_records = [json.loads(line) for line in f]
        
        # Consistent split
        random.seed(seed)
        random.shuffle(all_records)
        split_idx = int(len(all_records) * (1 - test_size))
        
        if split == 'train':
            self.records = all_records[:split_idx]
        else:
            self.records = all_records[split_idx:]
            
        print(f"Loaded {len(self.records)} samples for {split} split.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        audio_path = record["audio_path"]
        text = record["text"]
        
        # resampling to 16k for whisper
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # get spectrogram features
        input_features = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        
        # tokenize text
        labels = self.processor.tokenizer(text).input_ids
        
        return {"input_features": input_features, "labels": labels}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def main():
    # Load processor
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, 
        language=LANGUAGE, 
        task=TASK
    )
    
    # Initialize PyTorch Datasets
    train_dataset = WhisperHindiDataset(DATASET_PATH, processor, split='train')
    test_dataset = WhisperHindiDataset(DATASET_PATH, processor, split='test')

    # Load Model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = LANGUAGE.lower()
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
        
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=300, 
        gradient_checkpointing=True,
        fp16=True, 
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=25,
        eval_steps=25,
        logging_steps=10,
        report_to=[], # Disable tensorboard for simpler environment compatibility if needed
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False, # Important for custom PyTorch datasets with Trainer
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    
    print("Starting Model Training!")
    trainer.train(resume_from_checkpoint=True)
    
    # Save Custom model processor
    print("Saving the tuned model and processor...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("Training finished!")
    
if __name__ == "__main__":
    main()
