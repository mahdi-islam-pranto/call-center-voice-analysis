from google.cloud import speech_v2 as speech
from google.cloud.speech_v2.types import cloud_speech
from google.cloud import storage
from urllib.parse import urlparse
import os
from gcs_functions import download_transcription_and_save_to_txt, delete_files_from_gcs, delete_json_files_from_gcs, delete_files_from_local_with_prefix
import glob


# Transcribe long audio file using Chirp model
def transcribe_long_audio(gcs_uri, bucket_name, u_audio_name):
    # Set the correct region for Chirp_2 (e.g., "us" or "eu")
    location = "us-central1"  # Use "eu" for European Union

    # Initialize the client with the REGIONAL ENDPOINT
    client = speech.SpeechClient.from_service_account_file(
        'service-account.json',
        client_options={"api_endpoint": f"{location}-speech.googleapis.com"}
    )
    
    project_id = "woven-century-448009-r7"
    recognizer_id = "bangla-recognizer-2"
    parent = f"projects/{project_id}/locations/{location}"
    recognizer_name = f"{parent}/recognizers/{recognizer_id}"
    
    try:
        # Create/Get recognizer with Chirp model
        try:
            recognizer = client.get_recognizer(name=recognizer_name)
        except:
            recognizer = client.create_recognizer(
                parent=parent,
                recognizer_id=recognizer_id,
                recognizer={
                    "language_codes": ["bn-BD"],
                    "model": "chirp_2",
                    "default_recognition_config": {
                        "auto_decoding_config": {},
                        "features": {
                            "enable_automatic_punctuation": True,
                            "enable_word_time_offsets": True
                        }
                    }
                }
            )

        output_dir = "transcription_results/"
        # Output configuration
        output_config = cloud_speech.RecognitionOutputConfig(
            gcs_output_config=cloud_speech.GcsOutputConfig(
                uri=f"gs://{bucket_name}/{output_dir}"
            )
        )

        # Configure the request for Chirp model and transcribe
        request = cloud_speech.BatchRecognizeRequest(
            recognizer=recognizer_name,
            config=cloud_speech.RecognitionConfig(
                auto_decoding_config={},
                language_codes=["bn-BD"],
                model="chirp_2",
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=True,
                    # enable_spoken_punctuation=True
                )
            ),
            files=[{"uri": gcs_uri}],
            recognition_output_config=output_config
        )

        #print("Processing audio with Chirp model...")
        operation = client.batch_recognize(request=request)
        operation.result(timeout=3600)

        parsed_uri = urlparse(gcs_uri)
        base_name = os.path.splitext(os.path.basename(parsed_uri.path))[0]
        output_txt_filename = f"{base_name}_chirp2_transcript.txt"

        # Download and save the transcription
        fetch_final_transcript = download_transcription_and_save_to_txt(bucket_name, output_dir, output_txt_filename)

        # Delete the audio file and all JSON files in the transcription_results folder
        audio_file_path = parsed_uri.path.lstrip("/")  # Remove leading slash
        delete_files_from_gcs(bucket_name, [audio_file_path])  # Delete audio file
        delete_json_files_from_gcs(bucket_name, [output_dir], u_audio_name)  # Delete all JSON files in the folder
        folder_path = "/var/www/html/ai-projects/ihelp-crm" # Delete transcript fro local server
        delete_files_from_local_with_prefix(folder_path,u_audio_name)

        local_json_files = glob.glob(f"{u_audio_name}*.txt")
    
        if local_json_files:
            for file_path in local_json_files:
                os.remove(file_path)

        #return fetch
        return fetch_final_transcript

    except Exception as e:
        return {"error": f"Transcription error: {str(e)}"}