from pydantic import BaseModel
from typing import List
# async
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Query, Form, File, UploadFile
# import all functions
from process_single_audio import process_single_audio



# pydantic class for API input
class TranscriptionRequest(BaseModel):
    paths: List[str]
    summary_types: List[str]

# create thread pool
executor = ThreadPoolExecutor(max_workers=5)

app = FastAPI()

#get request
@app.post("/agent-performance")
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
    
    
    # sample audio  urls
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-093750_BUTTERFL_1_Ban_ECO_Plus_Shormi_01778952670-all.mp3",
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-091257_BUTTERFL_0_Ban_Talk_to_Cus_RP_Shormi_09638651762-all.mp3",
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-094725_OUTBOUND__Shormi_01718599694-all.mp3"
    