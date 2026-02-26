from google.cloud import storage
import os 
import json
import glob

# upload audio file to Google Storage
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client.from_service_account_json('service-account.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    #print(f"File {source_file_name} uploaded to GCS as {destination_blob_name}.")

# Exitiong file delete from Google Storage
def delete_files_from_gcs(bucket_name, folder_path):
    """Deletes all files in a folder from Google Cloud Storage."""
    storage_client = storage.Client.from_service_account_json('service-account.json')
    bucket = storage_client.bucket(bucket_name)
    
    # List all files in the folder
    blobs = bucket.list_blobs(prefix=folder_path)
    
    # Delete each file
    for blob in blobs:
        blob.delete()
        #print(f"Deleted file from GCS: gs://{bucket_name}/{blob.name}")

def delete_json_files_from_gcs(bucket_name, folder_path, u_audio_name):

    storage_client = storage.Client.from_service_account_json('service-account.json')
    bucket = storage_client.bucket(bucket_name)
    
    # List all files in the specified folder
    blobs = bucket.list_blobs(prefix=folder_path)

    # Define the prefix pattern
    prefix_pattern = f"transcription_results/{u_audio_name}_transcript_"

    for blob in blobs:
        if blob.name.startswith(prefix_pattern):
            blob.delete()
    else:
        return f"No matching file found for prefix: {prefix_pattern}"
    

def delete_files_from_local_with_prefix(folder_path, prefix):
    # Construct the search pattern to match files that start with the prefix
    pattern = os.path.join(folder_path, f"{prefix}*")
    
    # Use glob to find all files that match the pattern
    files_to_delete = glob.glob(pattern)
    
    # Check if any files were found
    if files_to_delete:
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                return 0
            except Exception as e:
                return 0
    else:
        return 0
    
   
# Download specific JSON file from GCS and save clean transcript to TXT locally
def download_transcription_and_save_to_txt(bucket_name, gcs_json_dir, output_txt_filename):
    """Downloads specific JSON result from GCS and saves clean transcript to TXT"""
    storage_client = storage.Client.from_service_account_json('service-account.json')
    bucket = storage_client.bucket(bucket_name)
    
    # Ensure the GCS directory path ends with a slash
    if not gcs_json_dir.endswith("/"):
        gcs_json_dir += "/"
    
    # List all JSON files in the GCS directory
    blobs = list(bucket.list_blobs(prefix=gcs_json_dir))
    if not blobs:
        #print(f"No JSON files found in gs://{bucket_name}/{gcs_json_dir}")
        return

    # Find the first JSON file in the directory
    json_blob = None
    for blob in blobs:
        if blob.name.endswith(".json"):
            json_blob = blob
            break
    
    if not json_blob:
        #print(f"No JSON files found in gs://{bucket_name}/{gcs_json_dir}")
        return

    # Download the JSON file locally
    local_json_path = os.path.basename(json_blob.name)
    json_blob.download_to_filename(local_json_path)
    #print(f"Downloaded JSON file: {local_json_path}")
    

    full_transcript = []   # final transcriptions
    
    try:
        with open(local_json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            
            for result in data.get("results", []):
                for alternative in result.get("alternatives", []):
                    transcript = alternative.get("transcript", "").strip()
                    if transcript:
                        full_transcript.append(transcript)
        
        # Join transcripts while maintaining order
        final_transcript = " ".join(full_transcript)
        
        # Clean up punctuation using regex
        import re
        final_transcript = re.sub(r'\s+([.,!?])', r'\1', final_transcript)
        final_transcript = re.sub(r'\s+', ' ', final_transcript)
        
        # Capitalize first letter and add final punctuation
        if final_transcript:
            final_transcript = final_transcript[0].upper() + final_transcript[1:]
            if final_transcript[-1] not in {'.', '!', '?'}:
                final_transcript += '.'
        else:
            final_transcript = "No transcription found."
        
        # Save to text file
        with open(output_txt_filename, "w", encoding="utf-8") as txt_file:
            txt_file.write(final_transcript)
            
        #print(f"Clean transcription saved to {output_txt_filename}")
        
        # Clean up: Delete the downloaded JSON file
        os.remove(local_json_path)
        

        return final_transcript

    except Exception as e:
        #print(f"Error processing JSON file: {e}")
        return {"error": f"Error processing JSON file: {e}"}

    finally:
        """ # Clean up: Delete the downloaded JSON file
        if os.path.exists(local_json_path):
            os.remove(local_json_path)
            #print(f"Deleted local JSON file: {local_json_path}") """