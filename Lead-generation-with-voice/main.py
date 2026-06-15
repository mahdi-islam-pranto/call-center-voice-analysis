from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
import base64
from typing import Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# initialize the FastAPI app
app = FastAPI(
    title= "Lead Finder API",
    description= "API to find potential leads based on user input.",
    version= "1.0.0"
)


# download audio file from the given url and store it in a temporary location, then return the file
def download_audio(url: str):
    try:
        # Stream the response to handle large files efficiently
        with requests.get(url, stream=True, verify=False) as response:
            # Check if the request was successful
            response.raise_for_status() 
            
            # current working directory
            current_dir = os.getcwd()
            
            print(f"current working directory: {current_dir}")
            
            # save the file in the current working directory with a temporary name
            temp_file_path = os.path.join(current_dir, "temp_audio_file.mp3")
            print(f"Saving audio to: {temp_file_path}")
            # Open local file in "write binary" mode
            with open(temp_file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        print(f"Success: Saved to {temp_file_path}")
        # return the audio file
        return temp_file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading audio: {e}")


SYSTEM_PROMPT = """
    You are an AI Lead Qualification Analyst for a courier and logistics company.

Your job is to analyze call transcripts between a call center agent and a customer and determine whether the customer is a potential business lead.

A potential lead is a customer who shows interest in any logistics, courier, shipping, export/import, business account, pricing inquiry, registration, quotation, or shipment-related service.

Analyze the entire conversation context, not just exact keywords.

The conversation may contain both Bangla and English languages.

You must identify:

Customer intent
Lead qualification status
Lead confidence score
Relevant keywords or phrases found
Important business requirements mentioned by the customer


Lead Indicators:

English: 
open account, 
new account
start shipping
want to ship
corporate account
business account
regular shipment
bulk shipment
registration
sign up
rate
price
charge
quotation
quote
proforma
discount
special rate
offer
rate card
rate sheet
compare rate
cheaper option
send abroad
international shipment
overseas
export
import
garments export
RMG shipment
document courier
urgent document
contract
agreement
MOU

Bangla:

অ্যাকাউন্ট খুলতে চাই
নতুন অ্যাকাউন্ট
শিপিং শুরু করতে চাই
পাঠাতে চাই
কর্পোরেট অ্যাকাউন্ট
ব্যবসায়িক অ্যাকাউন্ট
নিয়মিত পার্সেল
বাল্ক শিপমেন্ট
রেজিস্ট্রেশন করতে চাই
সাইন আপ
রেট
দাম
চার্জ
মূল্য
কত টাকা লাগবে
কত খরচ
কোটেশন
দরপত্র
ডিসকাউন্ট
বিশেষ রেট
অফার
রেট কার্ড
রেট লিস্ট
তুলনামূলক রেট
সস্তা অপশন
বিদেশে পাঠাতে চাই
আমেরিকায় পাঠাতে চাই
UK তে পাঠাতে চাই
আন্তর্জাতিক শিপমেন্ট
বিদেশে ডেলিভারি
এক্সপোর্ট
ইমপোর্ট
গার্মেন্টস পাঠাতে চাই
পোশাক রপ্তানি
ডকুমেন্ট পাঠাতে চাই
জরুরি কাগজ
চুক্তি করতে চাই
এমওইউ
এগ্রিমেন্ট

Important Rules:

Do not rely only on exact keyword matching.
Understand intent and business context.
If customer asks about rates, registration, business accounts, shipping process, quotations, export/import, or regular shipment needs, consider it a positive lead signal.
If the customer is merely seeking support for an existing shipment, tracking information, complaints, delivery status, or unrelated issues, do not classify as a lead unless new business intent is present.
Give higher confidence when multiple lead signals are found.
Give lower confidence when signals are weak or unclear.
    """



# pydantic input model for the API
class PotentialLeadRequest(BaseModel):
    audio_url: str
    
# pydantic output model for the API
class PotentialLeadResponse(BaseModel):
    potential_lead: bool
    keywords_matched: Optional[list[str]]
    justification: str
    lead_status: Literal["Strong Lead", "Potential Lead", "Weak Interest", "Not a Lead"]



# # download audio file from the given URL 
@app.post("/find_lead")
async def find_potential_lead(request: PotentialLeadRequest) -> PotentialLeadResponse:
    # download the audio file from the provided url
    audiofile_path = await download_audio(request.audio_url)
    
    # process the audio file
    # read and encode the audio file in base64
    with open(audiofile_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")
        
    # build the prompt for the model
    


# test download audio function
# audio_url = "https://103.204.81.3//RECORDINGS/MP3/20260615-101240_FEDEX_TN_FEDEX_zahidul_8801713351410-all.mp3"

# download_audio(audio_url)