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
from scipy.spatial.distance import cosine
import time
from collections import defaultdict

# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/pranavsinghpundir/voice_identification_system/.env")

# --- Constants ---
HF_TOKEN = os.environ.get("HF_TOKEN")
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_EMBEDDING_MODEL = "pyannote/embedding"
SIMILARITY_THRESHOLD = 0.55  # Adjust this value based on testing (0.0 to 1.0)
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

def load_salesperson_embeddings(embeddings_dir):
    """
    Loads all salesperson embeddings from the specified directory.
    """
    print("Loading salesperson embeddings...")
    embeddings = {}
    for file_name in os.listdir(embeddings_dir):
        if file_name.endswith(".npy"):
            salesperson_id = os.path.splitext(file_name)[0]
            embedding_path = os.path.join(embeddings_dir, file_name)
            embeddings[salesperson_id] = np.load(embedding_path)
            print(f"Loaded embedding for: {salesperson_id}")
    return embeddings

def identify_speaker_in_audio(audio_path, diarization_pipeline, embedding_extractor, salesperson_embeddings):
    """
    Identifies a speaker in a single audio file by comparing them to known salesperson embeddings.
    Returns the identified speaker ID or None if no match.
    """
    diarization_result = diarization_pipeline(audio_path)

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, -1.0, {}

    # Group segments by speaker_id from diarization_result
    speaker_segments_waveforms = defaultdict(list)
    for segment, _, speaker_id in diarization_result.itertracks(yield_label=True):
        segment_waveform = waveform[:, int(segment.start * 16000):int(segment.end * 16000)]
        if segment_waveform.shape[1] > 24000: # Min length of 1.5s (24000 samples)
            speaker_segments_waveforms[speaker_id].append(segment_waveform)

    for speaker_id_in_audio, segment_waveforms in speaker_segments_waveforms.items():
        if not segment_waveforms:
            continue

        # Extract embedding for this speaker_id_in_audio
        embeddings_for_this_speaker = []
        for seg_wf in segment_waveforms:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_f:
                sf.write(temp_f.name, seg_wf.squeeze().numpy(), 16000)
                embedding = embedding_extractor(temp_f.name)
                embeddings_for_this_speaker.append(embedding)

        if not embeddings_for_this_speaker:
            continue

        query_embedding = np.mean(np.array(embeddings_for_this_speaker), axis=0)

        current_speaker_similarities = {}
        current_speaker_best_similarity = -1.0
        current_speaker_best_match_id = None

        for salesperson_id, reference_embedding in salesperson_embeddings.items():
            similarity = 1 - cosine(query_embedding, reference_embedding)
            current_speaker_similarities[salesperson_id] = similarity
            if similarity > current_speaker_best_similarity:
                current_speaker_best_similarity = similarity
                current_speaker_best_match_id = salesperson_id

        # Update overall best match if this speaker_id_in_audio has a better match
        if current_speaker_best_similarity > overall_best_similarity:
            overall_best_similarity = current_speaker_best_similarity
            overall_best_match_id = current_speaker_best_match_id
            overall_all_similarities = current_speaker_similarities # Store similarities for the overall best match

    if overall_best_similarity > SIMILARITY_THRESHOLD:
        return overall_best_match_id, overall_best_similarity, overall_all_similarities
    else:
        return None, overall_best_similarity, overall_all_similarities # Return None if no match above threshold

def main():
    parser = argparse.ArgumentParser(description="Benchmark salesperson identification.")
    parser.add_argument("--testing_path_file", type=str, default=os.path.join(DATASET_BASE_PATH, "TestingAudio_Path.txt"), help="Path to the testing audio paths file.")
    args = parser.parse_args()

    output_dir_base = "/Users/pranavsinghpundir/voice_identification_system"
    embeddings_dir = os.path.join(output_dir_base, "salesperson_embeddings")

    salesperson_embeddings = load_salesperson_embeddings(embeddings_dir)
    if not salesperson_embeddings:
        print("No salesperson embeddings found. Please enroll speakers first.")
        return

    enrolled_speaker_ids = set(salesperson_embeddings.keys())
    print(f"Benchmarking with {len(enrolled_speaker_ids)} enrolled speakers.")
    print(f"Enrolled speaker IDs: {enrolled_speaker_ids}")

    test_audio_files = []
    with open(args.testing_path_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract speaker name from the test audio path (e.g., Aadiksha_6.wav -> Aadiksha)
            file_name_parts = os.path.splitext(line)[0].split('_')
            ground_truth_speaker_name = file_name_parts[0] 

            # Find the corresponding enrolled speaker ID (e.g., Aadiksha-007)
            ground_truth_speaker_id = None
            for enrolled_id in enrolled_speaker_ids:
                if enrolled_id.startswith(ground_truth_speaker_name + "-"):
                    ground_truth_speaker_id = enrolled_id
                    break
            
            if ground_truth_speaker_id and ground_truth_speaker_id in enrolled_speaker_ids:
                full_audio_path = os.path.join(DATASET_BASE_PATH, "TestingAudio", line)
                test_audio_files.append({
                    "path": full_audio_path,
                    "ground_truth_id": ground_truth_speaker_id
                })