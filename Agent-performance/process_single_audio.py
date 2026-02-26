import uuid
import tempfile
import os
from download_audio import download_audio_from_url
from gcs_functions import upload_to_gcs, delete_files_from_local_with_prefix
from transcribe_audio import transcribe_long_audio



# process single audio (download audio, upload to gcs, transcribe, delete audio and json files, summarize)
def process_single_audio(path, summary_types):
    # debug print
    print("Processing single audio file...")
    
    # Generate a unique name for the audio file
    unique_audio_name = str(uuid.uuid4())
        
    input_file = path

    bucket_name = "bangla_audio_files"
    #destination_blob_name = "call_files/call_recordings/transcripted_output.mp3"
    destination_blob_name = f"call_files/call_recordings/{unique_audio_name}.mp3"
        
    # Local file path where the downloaded file will be saved (for linux)
    # local_file_path = f"/tmp/{u_audio_name}.mp3"
        
    temp_dir = tempfile.gettempdir()
    local_file_path = os.path.join(temp_dir, f"{unique_audio_name}.mp3")

    # debug
    # print("File exists after download:", os.path.exists(local_file_path))
    # print("File size:", os.path.getsize(local_file_path) if os.path.exists(local_file_path)else "No file")
    # Download the audio from the URL
    if not download_audio_from_url(input_file, local_file_path):
        return {"error": "Failed to download the audio file from the provided URL."}
    # debug
    print(f"Downloaded audio file to {local_file_path}")
        
        
    upload_to_gcs(bucket_name, local_file_path, destination_blob_name)
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    response_text = transcribe_long_audio(gcs_uri, bucket_name, unique_audio_name)
    
    
    folder_path = "/tmp" 
    # Delete transcript fro local server
    delete_files_from_local_with_prefix(folder_path,unique_audio_name)
    
    return response_text