import uuid
from google.cloud import speech_v2 as speech
from google.cloud.speech_v2.types import cloud_speech
from google.cloud import storage
import json
import os
from urllib.parse import urlparse
from decouple import config
from pydantic import BaseModel
from typing import List
import fnmatch
import glob
import tempfile
import requests
# async
import asyncio
from concurrent.futures import ThreadPoolExecutor

SECRET_KEY = config('OPENAI_API_KEY')

from fastapi import FastAPI, Query, Form, File, UploadFile
app = FastAPI()

# openai model generate
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)

# prompt dynamic template initialization
from langchain_core.prompts import ChatPromptTemplate

# Function to download the audio file from the API URL and save it locally
""" def download_audio_from_url(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, "wb") as file:
            file.write(response.content)
        return True
    else:
        return False """

# Function to download the audio file from the URL and save it locally
import requests

def download_audio_from_url(url, local_path):
    try:
        response = requests.get(url, stream=True, verify=False)
        # debug
        # print("Status Code:", response.status_code)
        # print("Headers:", response.headers)
        # print("Content-Type:", response.headers.get("Content-Type"))

        if response.status_code == 200:
            with open(local_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

            print("Saved file exists?", os.path.exists(local_path))
            return True
        else:
            print("Failed with status:", response.status_code)
            return False

    except Exception as e:
        print("Download error:", e)
        return False

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

        # return {
        #         "transcript": response_text,
        #     }

        # system_template = "Summarize this bangla speech as a text point format. give me the response in Bangla and English. In your response, include the complain type of the customer"

    system_prompt = f"""
You are a professional QA analyst for a customer support call center.

Analyze the following Bangla transcript carefully and produce a structured analytical summary.

Rules:
- Write the response strictly in Bangla.
- Do not invent information.
- Base all conclusions strictly on transcript content.
- Keep it concise but analytically rich.
- Use professional language.

You must include analysis of: {summary_types}

Required Structure:

### ১. কলের সারাংশ:
- মূল আলোচনার বিষয়
- গ্রাহকের অনুরোধ বা সমস্যা
- এজেন্টের প্রতিক্রিয়া

### ২. গ্রাহক বিশ্লেষণ:
- গ্রাহকের উদ্দেশ্য
- সমস্যা বা অভিযোগের ধরন
- কথোপকথনের আবেগ বা টোন

### ৩. অতিরিক্ত পর্যবেক্ষণ:
- গ্রাহক সন্তুষ্ট ছিল কি না
- ভবিষ্যৎ ঝুঁকি (যদি থাকে)
"""
    prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "Here is the bangla call transcript: {text}")]
        )

        #invoke the prompt_template
    prompt = prompt_template.invoke({"text": response_text})

        #pass the template to the ai model

    response = model.invoke(prompt)

    folder_path = "/tmp" # Delete transcript fro local server
    delete_files_from_local_with_prefix(folder_path,unique_audio_name)
    
    return response.content
        

# pydantic class for API input
class TranscriptionRequest(BaseModel):
    paths: List[str]
    summary_types: List[str]

# create thread pool
executor = ThreadPoolExecutor(max_workers=5)

#get request
@app.post("/transcript")
async def api(request: TranscriptionRequest):
    # get event loop
    loop = asyncio.get_event_loop()
    
    # process each audio file in parallel
    tasks = [
        loop.run_in_executor(
            executor,
            process_single_audio,
            path,
            request.summary_types
        )
        for path in request.paths
    ]
    
    # wait for all tasks to complete
    
    results = await asyncio.gather(*tasks)
    
    return {
        "response_summary": results,
        "transcript_bearer": "Google Cloud Speech-to-Text",
        "summary_bearer": "Open AI",
    }
    
    
    
    # try:
    #     # debug
    #     print(f"API called with all parameters: {request.path}, {request.summary_types}")
    #     summary_type_string = ", ".join(request.summary_types)

    #     #input_file ="https://103.106.118.172/RECORDINGS/MP3/20250111-144224_HISHABEE__100171_01751220072-all.mp3" 
        
    #     # process single audio
    #     response_text = process_single_audio(request.path[0], summary_type_string)
        
    #     return {
    #             "response_summary": response_text,
    #             "transcript_bearer": "Google Cloud Speech-to-Text",
    #             "summary_bearer": "Open AI",
    #         }

    # except Exception as e:
    #     # Catch any errors and return a meaningful message
    #     return {"error": str(e)}