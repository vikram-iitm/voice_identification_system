
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
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/pranavsinghpundir/voice_identification_system/.env")

# --- Constants ---
HF_TOKEN = os.environ.get("HF_TOKEN")
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_EMBEDDING_MODEL = "pyannote/embedding"

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
    for segment, _, speaker_id in tqdm(diarization_result.itertracks(yield_label=True), desc="Extracting Embeddings"):
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
    parser = argparse.ArgumentParser(description="Enroll a salesperson and create a voice print.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the enrollment audio file.")
    parser.add_argument("--salesperson_id", type=str, required=True, help="Unique ID for the salesperson.")
    args = parser.parse_args()

    output_dir_base = "/Users/pranavsinghpundir/voice_identification_system"
    embeddings_dir = os.path.join(output_dir_base, "salesperson_embeddings")

    diarization_pipeline = get_diarization_pipeline()
    embedding_extractor = get_embedding_extractor()

    all_embeddings = []
    audio_files_to_process = []

    if os.path.isdir(args.audio_path):
        for root, _, files in os.walk(args.audio_path):
            for file in files:
                if file.endswith(".wav"):
                    audio_files_to_process.append(os.path.join(root, file))
    elif os.path.isfile(args.audio_path) and args.audio_path.endswith(".wav"):
        audio_files_to_process.append(args.audio_path)
    else:
        print(f"Invalid audio_path: {args.audio_path}. Must be a .wav file or a directory containing .wav files.")
        return

    if not audio_files_to_process:
        print(f"No .wav files found in {args.audio_path}.")
        return

    for audio_file in tqdm(audio_files_to_process, desc=f"Processing audio files for {args.salesperson_id}"):
        try:
            diarization_result = diarization_pipeline(audio_file)
            dominant_speaker = get_dominant_speaker(diarization_result)

            if dominant_speaker:
                waveform, sample_rate = torchaudio.load(audio_file)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                embeddings = []
                for segment, _, speaker_id in diarization_result.itertracks(yield_label=True):
                    if speaker_id == dominant_speaker:
                        segment_waveform = waveform[:, int(segment.start * 16000):int(segment.end * 16000)]
                        if segment_waveform.shape[1] > 24000:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_f:
                                sf.write(temp_f.name, segment_waveform.squeeze().numpy(), 16000)
                                embedding = embedding_extractor(temp_f.name)
                                embeddings.append(embedding)
                if embeddings:
                    all_embeddings.extend(embeddings)
            else:
                print(f"Could not identify a dominant speaker in {audio_file}. Skipping.")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

    if not all_embeddings:
        print(f"No valid embeddings extracted for {args.salesperson_id}. Cannot create voice print.")
        return

    reference_embedding = np.mean(np.array(all_embeddings), axis=0)
    embedding_path = os.path.join(embeddings_dir, f"{args.salesperson_id}.npy")
    np.save(embedding_path, reference_embedding)
    print(f"Voice print saved to: {embedding_path}")

if __name__ == "__main__":
    main()
