import argparse
import os
import numpy as np
import torch
import torchaudio
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
import soundfile as sf
import tempfile
from dotenv import load_dotenv
from collections import defaultdict
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/pranavsinghpundir/voice_identification_system/.env")

# --- Constants ---
HF_TOKEN = os.environ.get("HF_TOKEN")
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_EMBEDDING_MODEL = "pyannote/embedding"
DATASET_BASE_PATH = "/Users/pranavsinghpundir/Downloads/hindi-dataset"

def get_diarization_pipeline():
    """
    Initializes the pyannote.audio diarization pipeline.
    """
    if not HF_TOKEN:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    try:
        pipeline = Pipeline.from_pretrained(PYANNOTE_DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        print("Diarization pipeline loaded successfully.")
        return pipeline
    except Exception as e:
        print(f"Error loading pyannote.audio diarization pipeline: {e}")
        exit()

def get_embedding_extractor():
    """
    Initializes the pyannote.audio embedding model.
    """
    if not HF_TOKEN:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    try:
        model = Model.from_pretrained(PYANNOTE_EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
        extractor = Inference(model, window="whole")
        print("Embedding extractor loaded successfully.")
        return extractor
    except Exception as e:
        print(f"Error loading pyannote.audio embedding model: {e}")
        exit()

def get_dominant_speaker(diarization_result):
    """
    Identifies the dominant speaker from a diarization result.
    """
    speaker_times = {}
    for segment, _, speaker_id in diarization_result.itertracks(yield_label=True):
        if speaker_id not in speaker_times:
            speaker_times[speaker_id] = 0.0
        speaker_times[speaker_id] += segment.duration

    if not speaker_times:
        return None

    dominant_speaker = max(speaker_times, key=speaker_times.get)
    print(f"Dominant speaker identified: {dominant_speaker}")
    return dominant_speaker

def create_voice_print(audio_path, diarization_result, dominant_speaker, embedding_extractor, output_dir, speaker_id_for_filename):
    """
    Creates a voice print for the dominant speaker.
    """
    print("Creating voice print...")
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Ensure audio is mono for the embedding model
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    embeddings = []
    for segment, _, speaker_id in diarization_result.itertracks(yield_label=True):
        if speaker_id == dominant_speaker:
            segment_waveform = waveform[:, int(segment.start * 16000):int(segment.end * 16000)]
            # Ensure the segment is long enough for embedding (1.5s = 24000 samples at 16kHz)
            if segment_waveform.shape[1] > 24000: # Min length of 1.5s (24000 samples)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_f:
                    sf.write(temp_f.name, segment_waveform.squeeze().numpy(), 16000)
                    embedding = embedding_extractor(temp_f.name)
                    embeddings.append(embedding)

    if not embeddings:
        print("No segments found for the dominant speaker. Cannot create voice print.")
        return

    reference_embedding = np.mean(np.array(embeddings), axis=0)
    embedding_path = os.path.join(output_dir, f"{speaker_id_for_filename}.npy")
    np.save(embedding_path, reference_embedding)
    print(f"Voice print saved to: {embedding_path}")

def main():
    parser = argparse.ArgumentParser(description="Enroll speakers from a dataset and create voice prints.")
    parser.add_argument("--training_path_file", type=str, default=os.path.join(DATASET_BASE_PATH, "Voice_Samples_Training_Path.txt"), help="Path to the training audio paths file.")
    args = parser.parse_args()

    output_dir_base = "/Users/pranavsinghpundir/voice_identification_system"
    embeddings_dir = os.path.join(output_dir_base, "salesperson_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    # Parse training paths
    speaker_audio_files = defaultdict(list)
    with open(args.training_path_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('/')
            speaker_id = parts[0] # e.g., Abhay-001
            relative_audio_path = line
            speaker_audio_files[speaker_id].append(os.path.join(DATASET_BASE_PATH, "TrainingAudio", relative_audio_path))

    diarization_pipeline = get_diarization_pipeline()
    embedding_extractor = get_embedding_extractor()

    for speaker_id, audio_paths in tqdm(speaker_audio_files.items(), desc="Enrolling Speakers"):
        print(f"\nProcessing speaker: {speaker_id}")
        # Concatenate audio files for enrollment
        concatenated_waveform = []
        for ap in audio_paths:
            try:
                waveform, sample_rate = torchaudio.load(ap)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                concatenated_waveform.append(waveform)
            except Exception as e:
                print(f"Error loading {ap}: {e}")
                continue
        
        if not concatenated_waveform:
            print(f"Skipping {speaker_id}: No valid audio files found.")
            continue

        concatenated_waveform = torch.cat(concatenated_waveform, dim=1)
        
        # Save concatenated audio to a temporary file for diarization
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            sf.write(temp_audio_file.name, concatenated_waveform.squeeze().numpy(), 16000)
            
            # Diarize the concatenated audio
            diarization_result = diarization_pipeline(temp_audio_file.name)

            # Identify Dominant Speaker (should be the enrolled speaker)
            dominant_speaker = get_dominant_speaker(diarization_result)
            if not dominant_speaker:
                print(f"Could not identify dominant speaker for {speaker_id}.")
                continue

            # Create Voice Print
            create_voice_print(temp_audio_file.name, diarization_result, dominant_speaker, embedding_extractor, embeddings_dir, speaker_id)

if __name__ == "__main__":
    main()