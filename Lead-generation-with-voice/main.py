from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
import base64
from typing import Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
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

English keywords/phrases: 
open account, new account, start shipping, want to ship, corporate account, business account, regular shipment, bulk shipment, registration, sign up, rate, price, charge, how much does it cost, what is cost, quotation, quote, proforma, discount, special rate, offer, rate card, rate sheet, send abroad, send to USA, UK, international shipment, overseas, export, import, garments export, RMG shipment, document courier, urgent document, contract, agreement, MOU

Bangla keywords/phrases:
অ্যাকাউন্ট খুলতে চাই, নতুন অ্যাকাউন্ট, শিপিং শুরু করতে চাই, পাঠাতে চাই, কর্পোরেট অ্যাকাউন্ট, ব্যবসায়িক অ্যাকাউন্ট, নিয়মিত পার্সেল, বাল্ক শিপমেন্ট, রেজিস্ট্রেশন করতে চাই, সাইন আপ, রেট, দাম, চার্জ, কত টাকা লাগবে, কত খরচ, কোটেশন, দরপত্র, ডিসকাউন্ট, বিশেষ রেট, অফার আছে কি, রেট কার্ড পাঠান, রেট লিস্ট, তুলনামূলক রেট, সস্তা অপশন আছে কি, বিদেশে পাঠাতে চাই, আমেরিকায়, UK তে, আন্তর্জাতিক শিপমেন্ট, বিদেশে ডেলিভারি, এক্সপোর্ট, ইমপোর্ট করতে চাই, গার্মেন্টস পাঠাতে চাই, পোশাক রপ্তানি, ডকুমেন্ট পাঠাতে চাই, জরুরি কাগজ, চুক্তি করতে চাই, এমওইউ, এগ্রিমেন্ট

Important Rules:

Do not rely only on exact keyword matching.
Understand intent and business context.
If customer asks about rates, registration, business accounts, shipping process, quotations, export/import, or regular shipment needs, consider it a positive lead signal.
If the customer is merely seeking support for an existing shipment, tracking information, complaints, delivery status, or unrelated issues, do not classify as a lead unless new business intent is present.
If the customer talks about existing shipments, tracking, complaints, or delivery issues without new business intent, do not classify as a potential lead.
Give higher confidence when multiple lead signals are found.
Give lower confidence when signals are weak or unclear.
    """


# pydantic input model for the API
class PotentialLeadRequest(BaseModel):
    audio_url: str = Field(description="URL of the call recording audio file to be analyzed for lead qualification")
    
# pydantic output model for the API
class PotentialLeadResponse(BaseModel):
    is_potential_lead: bool = Field(description="Whether the customer is a potential lead or not")
    keywords_matched: Optional[list[str]] = Field(description="List of relevant keywords or phrases found in the conversation that indicate lead potential")
    justification: str = Field(description="A short explanation justifying the lead qualification decision based on the conversation analysis")
    lead_quality: Literal["Strong Lead", "Potential Lead", "Weak Interest", "Not a Lead"] = Field(description="A categorical assessment of the lead quality based on the analysis")
    tokens_used: int = Field(description="Number of tokens used in the analysis process")

# define llm
llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            # model_kwargs={
            #     "generation_config": {
            #         "thinking_config": {
            #         "thinking_budget": 0
            #             }
            #     }
            # }
        )

# # download audio file from the given URL 
@app.post("/find_lead")
async def find_potential_lead(request: PotentialLeadRequest) -> PotentialLeadResponse:
    # download the audio file from the provided url
    audiofile_path = download_audio(request.audio_url)
    
    # process the audio file
    # read and encode the audio file in base64
    with open(audiofile_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_b64 = base64.standard_b64encode(audio_bytes).decode("utf-8")
        
    # build the langchain messages for the model
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[
            {
                "type": "media",
                "data": audio_b64,
                "mime_type": "audio/mpeg",
            },
            {
                    "type": "text",
                    "text": """
                        Analyze the above call recording audio and determine whether the customer is a potential lead.

                        Instructions:

                        * Focus primarily on the CUSTOMER'S statements.
                        * Identify business intent.
                        * Detect both explicit and implicit lead signals.
                        * Consider Bangla and English phrases.
                        * Return keyword matches only if they are in my system prompt list.
                        * Provide a clear justification for your lead qualification decision.
                        * If the customer talks about existing shipments, tracking, complaints, or delivery issues without new business intent, do not classify as a potential lead.
                        * Return ONLY valid JSON.
                        * Do not include markdown formatting.
                        * Do not include explanations outside JSON.

                    """
            },
        ])
    ]
    
    try:
    
        structured_llm = llm.with_structured_output(PotentialLeadResponse, include_raw=True)
        result = structured_llm.invoke(messages)
        
        # delete the temporary audio file after processing
        # if os.path.exists(audiofile_path):
        #     os.remove(audiofile_path)
            
        parsed: PotentialLeadResponse = result["parsed"]
        raw_message = result["raw"]

        # Extract token usage from the raw AIMessage metadata
        usage_metadata = getattr(raw_message, "usage_metadata", {}) or {}
        tokens_used = usage_metadata.get("total_tokens", 0)
        
        
        return PotentialLeadResponse(
            is_potential_lead = parsed.is_potential_lead,
            keywords_matched = parsed.keywords_matched,
            justification = parsed.justification,
            lead_quality = parsed.lead_quality,
            tokens_used = tokens_used
        )
    except Exception as e:
        return PotentialLeadResponse(
            is_potential_lead = False,
            keywords_matched = [],
            justification = f"Error processing the audio: {str(e)}",
            lead_quality = "Not a Lead",
            tokens_used = 0
        )
    
    
    


# test download audio function
# audio_url = "https://103.204.81.3//RECORDINGS/MP3/20260615-101240_FEDEX_TN_FEDEX_zahidul_8801713351410-all.mp3"

# download_audio(audio_url)

# test (strong lead, price discussion)
# https://103.204.81.3/RECORDINGS/MP3/20260615-100751_FEDEX_TN_FEDEX_louis_8801862901607-all.mp3

# test (not a lead, irelevant query)
# https://103.204.81.3/RECORDINGS/MP3/20260615-091435_FEDEX_TN__louis_01712174109-all.mp3

# test (not a lead, traking delivery status)
# https://103.204.81.3/RECORDINGS/MP3/20260615-101255_FEDEX_TN_FEDEX_IMPORT_bijoy_8801923430572-all.mp3

