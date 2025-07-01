# Voice Identification System

This project implements a voice identification system, primarily focusing on identifying enrolled speakers within audio files. It utilizes `pyannote.audio` for speaker embedding extraction and a custom sliding-window approach for efficient identification without full diarization.

## Features

- **Speaker Enrollment:** Enroll known speakers by processing their audio samples and generating voice embeddings.
- **Efficient Identification:** Quickly identify enrolled speakers in new audio files using a sliding window technique, avoiding time-consuming full diarization.
- **Progress Bars:** Visual feedback during audio processing for both overall progress and individual file scanning.
- **Detailed Results:** Outputs identification results, including the identified salesperson, approximate timestamp of the match, and similarity score, to a text file.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vikram-iitm/voice_identification_system.git
    cd voice_identification_system
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv_voice_id
    source venv_voice_id/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face Token:** Ensure you have a Hugging Face token with read access. Set it as an environment variable named `HF_TOKEN` in a `.env` file in the project root. This token is required for `pyannote.audio` models.
    ```
    # .env file example
    HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN"
    ```

## Usage

### 1. Enroll Salespersons

(Instructions for `enroll_salesperson.py` would go here, based on its functionality. This README assumes `salesperson_embeddings` are already generated.)

### 2. Identify Salespersons in Audio Files

To identify enrolled salespersons in a directory of audio files, run the `identify_salesperson.py` script:

```bash
python scripts/identify_salesperson.py --audio_dir "/path/to/your/audio/files"
```

Replace `"/path/to/your/audio/files"` with the actual path to the directory containing the audio files you want to process.

Results will be saved to `output_identification/identification_results.txt`.

## Project Structure

- `audio_data/`: (Optional) Directory for storing raw audio data.
- `demucs_output/`: (Optional) Output from demucs processing.
- `hf_space_repo/`: (Ignored) Local Hugging Face Space repository.
- `output_identification/`: Stores identification results.
- `salesperson_embeddings/`: Stores pre-computed embeddings of enrolled salespersons.
- `scripts/`: Contains Python scripts for enrollment, identification, and benchmarking.
- `venv_voice_id/`: (Ignored) Python virtual environment.
- `.env`: (Ignored) Environment variables, including `HF_TOKEN`.
- `requirements.txt`: Project dependencies.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

(Specify your project's license here, e.g., MIT, Apache 2.0, etc.)
