import uuid
import tempfile
import os
from download_audio import download_audio_from_url
from gcs_functions import upload_to_gcs, delete_files_from_local_with_prefix
from transcribe_audio import transcribe_long_audio
from langchain_openai import ChatOpenAI
from decouple import config

# prompt dynamic template initialization
from langchain_core.prompts import ChatPromptTemplate

SECRET_KEY = config('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)


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

    folder_path = "/tmp" 
    # Delete transcript fro local server
    delete_files_from_local_with_prefix(folder_path,unique_audio_name)
    
    return response.content