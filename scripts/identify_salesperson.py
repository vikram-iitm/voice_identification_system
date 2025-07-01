import argparse
import os
import numpy as np
import torch
import torchaudio
from pyannote.audio import Model, Inference
from pyannote.core import Segment
import soundfile as sf
import tempfile
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
from tqdm import tqdm
import time

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/pranavsinghpundir/voice_identification_system/.env")

# --- Constants ---
HF_TOKEN = os.environ.get("HF_TOKEN")
PYANNOTE_EMBEDDING_MODEL = "pyannote/embedding"
SIMILARITY_THRESHOLD = 0.55  # Adjust this value based on testing (0.0 to 1.0)
WINDOW_SECONDS = 5  # Duration of the sliding window in seconds
STEP_SECONDS = 2.5    # How much the window slides forward each time

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

def load_salesperson_embeddings(embeddings_dir):
    """
    Loads all salesperson embeddings from the specified directory.
    """
    print(f"Loading salesperson embeddings from: {embeddings_dir}")
    embeddings = {}
    try:
        for file_name in os.listdir(embeddings_dir):
            if file_name.endswith(".npy"):
                salesperson_id = os.path.splitext(file_name)[0]
                embedding_path = os.path.join(embeddings_dir, file_name)
                embeddings[salesperson_id] = np.load(embedding_path)
                print(f"Loaded embedding for: {salesperson_id}")
    except FileNotFoundError:
        print(f"Error: Embeddings directory not found: {embeddings_dir}")
        return {}
    except Exception as e:
        print(f"An error occurred while loading embeddings: {e}")
        return {}
    return embeddings

def verify_speaker_in_chunk(chunk_embedding, salesperson_embeddings):
    """
    Compares a chunk's embedding to the salesperson embeddings.
    """
    for salesperson_id, reference_embedding in salesperson_embeddings.items():
        similarity = 1 - cosine(chunk_embedding, reference_embedding)
        if similarity > SIMILARITY_THRESHOLD:
            return salesperson_id, similarity
    return None, None

def write_result_to_file(output_file, audio_file, salesperson_id, timestamp, similarity, duration):
    """
    Writes the identification result to the specified output file.
    """
    with open(output_file, "a") as f:
        f.write(f"Audio File: {audio_file}\n")
        f.write(f"  - Identified as Salesperson: {salesperson_id}\n")
        f.write(f"  - Timestamp: ~{timestamp:.2f} seconds\n")
        f.write(f"  - Similarity: {similarity:.4f}\n")
        f.write(f"  - Processing Time: {duration:.2f} seconds\n\n")

def main():
    parser = argparse.ArgumentParser(description="Identify a salesperson in new audio files using a sliding window.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to the directory containing new audio files for identification.")
    args = parser.parse_args()

    output_dir_base = "/Users/pranavsinghpundir/voice_identification_system"
    embeddings_dir = os.path.join(output_dir_base, "salesperson_embeddings")
    output_file = os.path.join(output_dir_base, "output_identification", "identification_results.txt")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 1. Load enrolled salesperson embeddings
    salesperson_embeddings = load_salesperson_embeddings(embeddings_dir)
    if not salesperson_embeddings:
        print("No salesperson embeddings found. Please enroll a salesperson first.")
        return

    # 2. Get the embedding model
    embedding_extractor = get_embedding_extractor()

    # 3. Process each audio file in the directory
    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith(('.wav', '.mp3', '.flac', '.WAV'))]
    for audio_file in audio_files:
        audio_path = os.path.join(args.audio_dir, audio_file)
        print(f"\nProcessing: {audio_path}")
        start_time = time.time()

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            num_samples = waveform.shape[1]
            window_samples = int(WINDOW_SECONDS * 16000)
            step_samples = int(STEP_SECONDS * 16000)

            found_match = False
            with tqdm(total=num_samples, desc=f"Scanning {audio_file}", unit="samples") as pbar:
                for start_sample in range(0, num_samples - window_samples + 1, step_samples):
                    end_sample = start_sample + window_samples
                    segment = waveform[:, start_sample:end_sample]

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_f:
                        sf.write(temp_f.name, segment.squeeze().numpy(), 16000)
                        chunk_embedding = embedding_extractor(temp_f.name)
                    
                    salesperson_id, similarity = verify_speaker_in_chunk(chunk_embedding, salesperson_embeddings)
                    if salesperson_id:
                        timestamp = start_sample / 16000
                        duration = time.time() - start_time
                        print(f"\n*** MATCH FOUND in {audio_file} ***")
                        print(f"Identified as Salesperson {salesperson_id} at ~{timestamp:.2f} seconds (Similarity: {similarity:.4f}).\n")
                        write_result_to_file(output_file, audio_file, salesperson_id, timestamp, similarity, duration)
                        found_match = True
                        break # Stop processing this file after the first match
                    pbar.update(step_samples)

            if not found_match:
                duration = time.time() - start_time
                print(f"No match found for any salesperson in {audio_file}.")
                with open(output_file, "a") as f:
                    f.write(f"Audio File: {audio_file}\n")
                    f.write(f"  - No match found for any salesperson.\n")
                    f.write(f"  - Processing Time: {duration:.2f} seconds\n\n")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

if __name__ == "__main__":
    main()